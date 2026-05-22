"""
telephony/stt.py

Deepgram streaming STT with:
- Real-time word-by-word transcription
- Language detection (en/hi/ta) via fastText fallback
- End-of-speech signalling for VAD integration
- Latency measurement from audio-start to final transcript
"""

import asyncio
import time
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Callable

import websockets
import json

logger = logging.getLogger(__name__)

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"

# Language-specific model config
_LANG_CONFIG = {
    "en": {"model": "nova-2", "language": "en-IN"},
    "hi": {"model": "nova-2-general", "language": "hi"},
    "ta": {"model": "nova-2-general", "language": "ta"},
    # Auto-detect mode for first turn
    "auto": {"model": "nova-2-general", "detect_language": True},
}


@dataclass
class TranscriptResult:
    text: str
    language: str
    confidence: float
    is_final: bool
    duration_ms: float


class DeepgramSTTClient:
    """
    Streaming STT client for Deepgram.
    
    Usage:
        async with DeepgramSTTClient(api_key) as client:
            async for result in client.stream(audio_chunks, language="hi"):
                if result.is_final:
                    process(result.text)
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def stream(
        self,
        audio_source: AsyncGenerator[bytes, None],
        language: str = "auto",
        on_interim: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[TranscriptResult, None]:
        """
        Streams audio to Deepgram. Yields TranscriptResult for each utterance.
        - Interim results are passed to on_interim callback (for display/barge-in)
        - Final results are yielded when Deepgram signals speech_final
        """
        config = _LANG_CONFIG.get(language, _LANG_CONFIG["auto"])
        params = self._build_params(config)
        url = f"{DEEPGRAM_WS_URL}?{params}"

        t_start = time.monotonic()
        detected_language = language if language != "auto" else "en"

        async with websockets.connect(
            url,
            extra_headers={"Authorization": f"Token {self.api_key}"},
            ping_interval=5,
        ) as ws:
            result_queue: asyncio.Queue[Optional[TranscriptResult]] = asyncio.Queue()

            async def _send_audio():
                async for chunk in audio_source:
                    await ws.send(chunk)
                # Signal end of audio
                await ws.send(json.dumps({"type": "CloseStream"}))

            async def _receive():
                nonlocal detected_language
                async for raw in ws:
                    msg = json.loads(raw)

                    if msg.get("type") == "Results":
                        channel = msg.get("channel", {})
                        alts = channel.get("alternatives", [])
                        if not alts:
                            continue

                        transcript = alts[0].get("transcript", "").strip()
                        if not transcript:
                            continue

                        confidence = alts[0].get("confidence", 1.0)
                        is_final = msg.get("speech_final", False)

                        # Deepgram returns detected_language in metadata on first result
                        if "detected_language" in msg:
                            detected_language = _map_deepgram_lang(msg["detected_language"])

                        duration_ms = (time.monotonic() - t_start) * 1000

                        if on_interim and not is_final:
                            on_interim(transcript)

                        if is_final:
                            await result_queue.put(TranscriptResult(
                                text=transcript,
                                language=detected_language,
                                confidence=confidence,
                                is_final=True,
                                duration_ms=duration_ms,
                            ))

                    elif msg.get("type") == "Metadata":
                        logger.debug("Deepgram metadata: %s", msg)

                    elif msg.get("type") == "CloseStream":
                        break

                await result_queue.put(None)  # sentinel

            send_task = asyncio.create_task(_send_audio())
            recv_task = asyncio.create_task(_receive())

            try:
                while True:
                    result = await result_queue.get()
                    if result is None:
                        break
                    yield result
            finally:
                send_task.cancel()
                recv_task.cancel()

    def _build_params(self, config: dict) -> str:
        parts = [
            f"model={config['model']}",
            "encoding=linear16",
            "sample_rate=16000",
            "channels=1",
            "interim_results=true",
            "smart_format=true",
            "endpointing=300",  # 300ms silence = end of utterance
            "utterance_end_ms=1000",
        ]
        if "language" in config:
            parts.append(f"language={config['language']}")
        if config.get("detect_language"):
            parts.append("detect_language=true")
        return "&".join(parts)


def _map_deepgram_lang(dg_lang: str) -> str:
    """Maps Deepgram language codes to our internal codes."""
    mapping = {
        "en": "en", "en-IN": "en", "en-US": "en",
        "hi": "hi",
        "ta": "ta",
    }
    return mapping.get(dg_lang, "en")


# ─── FastText language detector (fallback / first-turn detection) ─────────────

class LanguageDetector:
    """
    Lightweight language detector for en/hi/ta.
    Uses fastText model; falls back to heuristic on import failure.
    """

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            import fasttext
            self._model = fasttext.load_model("models/lid.176.ftz")
            logger.info("fastText language model loaded")
        except Exception as e:
            logger.warning("fastText not available, using heuristic detector: %s", e)

    def detect(self, text: str) -> str:
        if self._model:
            labels, _ = self._model.predict(text.replace("\n", " "))
            lang = labels[0].replace("__label__", "")
            if lang in ("hi", "ta"):
                return lang
            return "en"

        # Heuristic: Tamil Unicode range U+0B80–U+0BFF, Hindi U+0900–U+097F
        for ch in text:
            cp = ord(ch)
            if 0x0B80 <= cp <= 0x0BFF:
                return "ta"
            if 0x0900 <= cp <= 0x097F:
                return "hi"
        return "en"
