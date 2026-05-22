"""
api/main.py

FastAPI application.
Endpoints:
  WS  /ws/call/{call_id}    — real-time voice call (inbound)
  POST /api/calls/outbound  — trigger outbound call (campaign)
  GET  /health              — liveness check
  GET  /metrics             — Prometheus metrics
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import anthropic
import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.orchestrator import AgentOrchestrator
from agent.tools import init_tools
from memory.session_store import SessionStore
from memory.session_store import LongTermMemory
from services.scheduling import SchedulingService
from services.patient import PatientService
from telephony.stt import DeepgramSTTClient, LanguageDetector
from telephony.tts import TTSStreamer
from telephony.vad import SileroVAD, VADPipeline
from workers.campaign import enqueue_outbound_call

logger = logging.getLogger(__name__)

# ─── App state (shared across requests) ───────────────────────────────────────

class AppState:
    pool: asyncpg.Pool = None
    redis: aioredis.Redis = None
    session_store: SessionStore = None
    long_term: LongTermMemory = None
    scheduling: SchedulingService = None
    patient: PatientService = None
    tts: TTSStreamer = None
    stt_client: DeepgramSTTClient = None
    lang_detector: LanguageDetector = None
    anthropic_client: anthropic.AsyncAnthropic = None
    vad: SileroVAD = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("Starting Voice Agent server...")

    state.pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"],
        min_size=5,
        max_size=20,
        command_timeout=10,
    )
    state.redis = aioredis.from_url(
        os.environ["REDIS_URL"],
        encoding="utf-8",
        decode_responses=True,
    )
    state.session_store = SessionStore(state.redis)
    state.long_term = LongTermMemory(state.pool)
    state.scheduling = SchedulingService(state.pool)
    state.patient = PatientService(state.pool)
    state.tts = TTSStreamer(
        elevenlabs_api_key=os.environ["ELEVENLABS_API_KEY"],
        azure_subscription_key=os.environ.get("AZURE_SPEECH_KEY", ""),
        azure_region=os.environ.get("AZURE_SPEECH_REGION", "centralindia"),
    )
    state.stt_client = DeepgramSTTClient(os.environ["DEEPGRAM_API_KEY"])
    state.lang_detector = LanguageDetector()
    state.anthropic_client = anthropic.AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )
    state.vad = SileroVAD()

    init_tools(state.scheduling, state.patient)

    logger.info("All services initialised.")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    await state.pool.close()
    await state.redis.aclose()
    logger.info("Server shut down cleanly.")


app = FastAPI(
    title="2Care Voice Agent",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health + metrics ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    redis_ok = await state.session_store.ping()
    try:
        await state.pool.fetchval("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "ok" if (redis_ok and db_ok) else "degraded",
        "redis": redis_ok,
        "db": db_ok,
    }


# ─── WebSocket: Inbound call ───────────────────────────────────────────────────

@app.websocket("/ws/call/{call_id}")
async def inbound_call(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint for a live call.
    
    Protocol:
      Client → Server: raw PCM16 audio bytes (16kHz mono)
      Server → Client: raw PCM16 audio bytes (TTS response)
      Server → Client: JSON control frames {"type": "transcript", "text": "..."}
    """
    await websocket.accept()
    logger.info("Call connected: %s", call_id)

    orchestrator = AgentOrchestrator(
        session_store=state.session_store,
        long_term_memory=state.long_term,
        tts_streamer=state.tts,
        anthropic_client=state.anthropic_client,
    )

    vad_pipeline = VADPipeline(
        vad=state.vad,
        on_speech_start=lambda: logger.debug("Speech started"),
        on_barge_in=lambda: state.tts.cancel(),
    )

    # Detect patient from call_id (e.g. lookup via Twilio call SID)
    patient_id = await state.patient.lookup_by_call_id(call_id)
    if patient_id:
        session = await state.session_store.get(call_id)
        if session:
            session.patient_id = patient_id
            await state.session_store.set(call_id, session)

    # Language preference from last session
    preferred_language = "en"
    if patient_id:
        preferred_language = await state.patient.get_language_preference(patient_id) or "en"

    import json as _json

    async def audio_source():
        """Yields raw audio bytes from WebSocket."""
        try:
            while True:
                data = await websocket.receive_bytes()
                yield data
        except WebSocketDisconnect:
            return

    try:
        async for utterance_audio in vad_pipeline.process_stream(audio_source()):
            # Feed utterance to STT
            t_stt_start = time.monotonic()
            transcript_text = ""
            detected_language = preferred_language

            async for result in state.stt_client.stream(
                _single_chunk_source(utterance_audio),
                language=preferred_language,
            ):
                if result.is_final:
                    transcript_text = result.text
                    detected_language = result.language
                    break

            if not transcript_text.strip():
                continue

            stt_ms = (time.monotonic() - t_stt_start) * 1000

            # Send transcript back to client (for UI display)
            await websocket.send_text(_json.dumps({
                "type": "transcript",
                "text": transcript_text,
                "language": detected_language,
            }))

            logger.info("Turn: call=%s lang=%s text=%r", call_id, detected_language, transcript_text)

            # Update language preference if changed
            if detected_language != preferred_language and patient_id:
                await state.patient.update_language_preference(patient_id, detected_language)
                preferred_language = detected_language

            # Run agent turn — stream TTS back
            vad_pipeline.set_tts_playing(True)
            try:
                async for audio_chunk in orchestrator.handle_turn(
                    call_id=call_id,
                    transcript=transcript_text,
                    language=detected_language,
                    stt_duration_ms=stt_ms,
                ):
                    await websocket.send_bytes(audio_chunk)
            finally:
                vad_pipeline.set_tts_playing(False)

    except WebSocketDisconnect:
        logger.info("Call disconnected: %s", call_id)
    except Exception as e:
        logger.exception("Call error %s: %s", call_id, e)
    finally:
        # Post-call: save interaction summary async
        asyncio.create_task(_save_call_summary(call_id, patient_id))


async def _save_call_summary(call_id: str, patient_id: str | None):
    """Generates and saves a post-call summary for long-term memory."""
    if not patient_id:
        return
    session = await state.session_store.get(call_id)
    if not session or session.turn_count < 2:
        return
    try:
        # Summarise conversation via a quick LLM call
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in session.conversation_history[-10:]
        )
        resp = await state.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarise this clinical appointment call in 1-2 sentences, "
                    f"noting outcome and any preferences mentioned:\n\n{history_text}"
                ),
            }],
        )
        summary = resp.content[0].text
        await state.long_term.save_interaction(
            patient_id=patient_id,
            call_id=call_id,
            summary=summary,
            language=session.language,
        )
    except Exception as e:
        logger.warning("Failed to save call summary for %s: %s", call_id, e)


async def _single_chunk_source(audio: bytes):
    """Wraps a single bytes object as an async generator for STT."""
    yield audio


# ─── REST: Outbound call ───────────────────────────────────────────────────────

class OutboundCallRequest(BaseModel):
    patient_id: str
    campaign_id: str
    appointment_id: str | None = None
    call_type: str = "reminder"   # reminder | followup | confirmation


@app.post("/api/calls/outbound")
async def trigger_outbound_call(req: OutboundCallRequest):
    """Enqueue an outbound call for a patient. Executed by Celery worker."""
    task = enqueue_outbound_call(
        patient_id=req.patient_id,
        campaign_id=req.campaign_id,
        appointment_id=req.appointment_id,
        call_type=req.call_type,
    )
    return {"task_id": task.id, "status": "queued"}


@app.get("/api/calls/{call_id}/latency")
async def get_call_latency(call_id: str):
    """Return latency metrics for a completed call (from JSONL log)."""
    # In production this would query a time-series DB or log aggregator
    return {"call_id": call_id, "message": "Query your log aggregator for LATENCY_REPORT entries"}
