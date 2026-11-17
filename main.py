"""
EstudIA - Backend
-----------------
Servidor pequeño que:
  1. Extrae el texto de PDF, DOCX o TXT.
  2. Pide a Claude que genere un cuestionario.
  3. Pide a Claude que evalúe la respuesta del usuario.
  4. Transcribe audio a texto (Whisper de OpenAI o Groq).

La app nunca ve tus claves de API: solo habla con este servidor.

Cuentas de usuario (opcional, con Supabase):
  - Si defines SUPABASE_URL, cada petición debe venir con la sesión del usuario
    (Authorization: Bearer <token>) y se aplican límites diarios por usuario.
  - Mientras PERMITIR_TOKEN_APP=true, también se acepta el APP_TOKEN antiguo
    (útil durante la transición).
"""

import io
import json
import os
import random
import re
import time

import httpx
from docx import Document
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

TRANSCRIBE_API_KEY = os.getenv("TRANSCRIBE_API_KEY", "")
TRANSCRIBE_BASE_URL = os.getenv("TRANSCRIBE_BASE_URL", "https://api.openai.com/v1")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")

APP_TOKEN = os.getenv("APP_TOKEN", "")
MAX_CHARS = int(os.getenv("MAX_CHARS", "60000"))

# --- Cuentas de usuario (Supabase)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
PERMITIR_TOKEN_APP = os.getenv("PERMITIR_TOKEN_APP", "true").lower() == "true"
ADMIN_EMAILS = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}

# Límites diarios por usuario (0 = sin límite)
LIMITES = {
    "extraer": int(os.getenv("LIMITE_EXTRAER_DIA", "20")),
    "cuestionario": int(os.getenv("LIMITE_CUESTIONARIOS_DIA", "15")),
    "evaluar": int(os.getenv("LIMITE_EVALUAR_DIA", "200")),
    "transcribir": int(os.getenv("LIMITE_TRANSCRIBIR_DIA", "400")),
}
NOMBRES_USO = {
    "extraer": "archivos subidos",
    "cuestionario": "cuestionarios generados",
    "evaluar": "respuestas corregidas",
    "transcribir": "respuestas por voz",
}

app = FastAPI(title="EstudIA backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------------------------------------------------------------- usuarios

_cache_sesiones: dict[str, tuple[float, dict]] = {}


def _cabeceras_supabase(clave: str) -> dict:
    cab = {"apikey": clave}
    if not clave.startswith("sb_"):  # claves antiguas (JWT) también van como Bearer
        cab["Authorization"] = f"Bearer {clave}"
    return cab


async def _usuario_desde_token(token: str) -> dict:
    ahora = time.time()
    guardado = _cache_sesiones.get(token)
    if guardado and guardado[0] > ahora:
        return guardado[1]
    async with httpx.AsyncClient(timeout=15) as cliente:
        r = await cliente.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"},
        )
    if r.status_code != 200:
        raise HTTPException(401, "Tu sesión expiró. Vuelve a iniciar sesión.")
    datos = r.json()
    usuario = {"id": datos["id"], "email": (datos.get("email") or "").lower()}
    if len(_cache_sesiones) > 2000:
        _cache_sesiones.clear()
    _cache_sesiones[token] = (ahora + 300, usuario)
    return usuario


async def verificar_usuario(
    authorization: str = Header(default=""), x_app_token: str = Header(default="")
) -> dict:
    """Devuelve el usuario que hace la petición, o rechaza la petición."""
    if SUPABASE_URL and authorization.lower().startswith("bearer "):
        return await _usuario_desde_token(authorization[7:].strip())

    # Modo antiguo / transición: token compartido de la app
    if not SUPABASE_URL or PERMITIR_TOKEN_APP:
        if not APP_TOKEN or x_app_token == APP_TOKEN:
            return {"id": None, "email": ""}
        raise HTTPException(401, "Token inválido")

    raise HTTPException(401, "Debes iniciar sesión.")


async def registrar_uso(usuario: dict, tipo: str):
    """Suma 1 al uso diario del usuario y rechaza si superó su límite."""
    if not usuario.get("id") or not SUPABASE_SERVICE_KEY:
        return
    try:
        async with httpx.AsyncClient(timeout=15) as cliente:
            r = await cliente.post(
                f"{SUPABASE_URL}/rest/v1/rpc/incrementar_uso",
                headers={**_cabeceras_supabase(SUPABASE_SERVICE_KEY), "Content-Type": "application/json"},
                json={"p_user": usuario["id"], "p_tipo": tipo},
            )
        cantidad = int(r.json()) if r.status_code == 200 else 0
    except Exception:  # noqa: BLE001  si falla el conteo, no bloqueamos al usuario
        return
    limite = LIMITES.get(tipo, 0)
    if limite and cantidad > limite and usuario.get("email") not in ADMIN_EMAILS:
        raise HTTPException(
            429,
            f"Llegaste al límite diario de {NOMBRES_USO.get(tipo, tipo)} ({limite}). Vuelve a intentarlo mañana.",
        )


# ---------------------------------------------------------------- utilidades


async def preguntar_a_claude(system: str, prompt: str, max_tokens: int = 4000) -> str:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "Falta ANTHROPIC_API_KEY en el servidor")
    async with httpx.AsyncClient(timeout=180) as cliente:
        r = await cliente.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Error de la IA: {r.text[:300]}")
    data = r.json()
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def extraer_json(texto: str):
    m = re.search(r"[\[{].*[\]}]", texto, re.S)
    if not m:
        raise HTTPException(502, "La IA no devolvió un JSON válido")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        raise HTTPException(502, "La IA devolvió un JSON con errores, intenta de nuevo")


# ---------------------------------------------------------------- endpoints


@app.get("/")
def salud():
    return {"ok": True, "servicio": "EstudIA", "cuentas": bool(SUPABASE_URL)}


@app.get("/yo")
async def yo(usuario: dict = Depends(verificar_usuario)):
    """Permite a la app comprobar que la sesión es válida."""
    return usuario


@app.post("/extraer")
async def extraer(archivo: UploadFile = File(...), usuario: dict = Depends(verificar_usuario)):
    await registrar_uso(usuario, "extraer")
    datos = await archivo.read()
    nombre = (archivo.filename or "").lower()
    try:
        if nombre.endswith(".pdf"):
            lector = PdfReader(io.BytesIO(datos))
            texto = "\n\n".join((p.extract_text() or "") for p in lector.pages)
        elif nombre.endswith(".docx"):
            doc = Document(io.BytesIO(datos))
            texto = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif nombre.endswith((".txt", ".md")):
            texto = datos.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(400, "Formato no soportado. Usa PDF, DOCX o TXT.")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"No se pudo leer el archivo: {e}")

    texto = re.sub(r"[ \t]+", " ", texto).strip()
    if len(texto) < 50:
        raise HTTPException(
            400, "El archivo casi no tiene texto. ¿Es un PDF escaneado (foto)? Esos aún no se soportan."
        )
    return {"texto": texto}


class PedidoCuestionario(BaseModel):
    texto: str
    cantidad: int = 10
    tipo: str = "alternativas"  # "alternativas" o "abiertas"


REGLAS_COMUNES = """- Cubre las ideas más importantes del material, de lo básico a lo más difícil.
- Redacta para que se entienda al escucharla (también se usará por voz mientras la persona maneja):
  sin tablas, símbolos raros ni "según la figura"."""


@app.post("/cuestionario")
async def cuestionario(p: PedidoCuestionario, usuario: dict = Depends(verificar_usuario)):
    await registrar_uso(usuario, "cuestionario")
    cantidad = max(3, min(p.cantidad, 30))
    abiertas = p.tipo == "abiertas"
    system = (
        "Eres un profesor experto que prepara cuestionarios de estudio en español. "
        "Respondes únicamente con JSON válido, sin texto adicional."
    )
    if abiertas:
        prompt = f"""Crea {cantidad} preguntas abiertas de estudio basadas SOLO en el material de abajo.

Reglas:
- Preguntas sin alternativas, que se puedan responder de forma oral en una o dos frases.
{REGLAS_COMUNES}
- "respuesta": respuesta correcta breve (una o dos frases).
- "explicacion": una frase que ayude a recordar o entender.

Devuelve SOLO este JSON:
[{{"pregunta": "...", "respuesta": "...", "explicacion": "..."}}]

MATERIAL:
<<<
{p.texto[:MAX_CHARS]}
>>>"""
    else:
        prompt = f"""Crea {cantidad} preguntas de selección múltiple basadas SOLO en el material de abajo.

Reglas:
- Cada pregunta tiene UNA alternativa correcta y TRES incorrectas.
- Las incorrectas deben ser creíbles (errores típicos), pero claramente incorrectas según el material.
- Alternativas breves (idealmente menos de 12 palabras), porque se leerán en voz alta.
- No uses "todas las anteriores" ni "ninguna de las anteriores".
{REGLAS_COMUNES}
- "explicacion": una o dos frases que expliquen por qué la correcta es la correcta.

Devuelve SOLO este JSON:
[{{"pregunta": "...", "correcta": "...", "incorrectas": ["...", "...", "..."], "explicacion": "..."}}]

MATERIAL:
<<<
{p.texto[:MAX_CHARS]}
>>>"""

    data = extraer_json(await preguntar_a_claude(system, prompt, max_tokens=8000))
    if isinstance(data, dict):
        data = data.get("preguntas", [])

    preguntas = []
    for q in data:
        if not isinstance(q, dict) or not q.get("pregunta"):
            continue
        if abiertas:
            if not q.get("respuesta"):
                continue
            preguntas.append({k: str(q.get(k, "")).strip() for k in ("pregunta", "respuesta", "explicacion")})
            continue
        if not q.get("correcta"):
            continue
        incorrectas = [str(x).strip() for x in (q.get("incorrectas") or []) if str(x).strip()][:3]
        if len(incorrectas) < 3:
            continue
        correcta = str(q["correcta"]).strip()
        alternativas = [correcta] + incorrectas
        random.shuffle(alternativas)  # la correcta queda en A, B, C o D al azar
        preguntas.append(
            {
                "pregunta": str(q["pregunta"]).strip(),
                "alternativas": alternativas,
                "correcta": alternativas.index(correcta),  # 0 = A, 1 = B, 2 = C, 3 = D
                "respuesta": correcta,
                "explicacion": str(q.get("explicacion", "")).strip(),
            }
        )
    return {"preguntas": preguntas}


class PedidoEvaluacion(BaseModel):
    pregunta: str
    respuesta_correcta: str
    respuesta_usuario: str


@app.post("/evaluar")
async def evaluar(p: PedidoEvaluacion, usuario: dict = Depends(verificar_usuario)):
    await registrar_uso(usuario, "evaluar")
    system = (
        "Eres un profesor amable que corrige respuestas de estudio. "
        "Respondes únicamente con JSON válido, sin texto adicional."
    )
    prompt = f"""Evalúa la respuesta de un estudiante.

La respuesta pudo ser dictada por voz y transcrita automáticamente: ignora errores de ortografía
o palabras mal transcritas. Considera "correcta" una respuesta que exprese la misma idea con otras palabras.
Usa "parcial" si tiene una parte importante pero le falta algo clave, e "incorrecta" si no corresponde.

Pregunta: {p.pregunta}
Respuesta esperada: {p.respuesta_correcta}
Respuesta del estudiante: {p.respuesta_usuario}

Devuelve SOLO este JSON:
{{"resultado": "correcta" | "parcial" | "incorrecta", "comentario": "..."}}

El comentario: máximo 2 frases cortas, en segunda persona, tono cercano, pensado para escucharse
en voz alta (sin listas, símbolos ni markdown). Si no es correcta, incluye brevemente la respuesta correcta."""
    data = extraer_json(await preguntar_a_claude(system, prompt, max_tokens=500))
    resultado = str(data.get("resultado", "incorrecta")).lower()
    if resultado not in ("correcta", "parcial", "incorrecta"):
        resultado = "incorrecta"
    return {"resultado": resultado, "comentario": str(data.get("comentario", "")).strip()}


@app.post("/transcribir")
async def transcribir(
    audio: UploadFile = File(...), prompt: str = Form(default=""), usuario: dict = Depends(verificar_usuario)
):
    await registrar_uso(usuario, "transcribir")
    if not TRANSCRIBE_API_KEY:
        raise HTTPException(500, "Falta TRANSCRIBE_API_KEY en el servidor")
    datos = await audio.read()
    campos = {"model": TRANSCRIBE_MODEL, "language": "es"}
    if prompt:
        campos["prompt"] = prompt  # ayuda a reconocer respuestas cortas como "la B"
    async with httpx.AsyncClient(timeout=90) as cliente:
        r = await cliente.post(
            f"{TRANSCRIBE_BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {TRANSCRIBE_API_KEY}"},
            data=campos,
            files={"file": (audio.filename or "respuesta.m4a", datos, audio.content_type or "audio/m4a")},
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Error al transcribir: {r.text[:300]}")
    return {"texto": r.json().get("text", "").strip()}
