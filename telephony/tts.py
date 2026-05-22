"""
telephony/tts.py

Streaming TTS with:
- ElevenLabs for English (lowest latency, best quality)
- Azure Neural for Hindi / Tamil (better language quality)
- First-chunk target: <80ms
- Barge-in: exposes cancel() to flush mid-stream

telephony/vad.py is inlined here for simplicity.
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

import httpx

logger = logging.getLogger(__name__)

# ─── Voice IDs / config ────────────────────────────────────────────────────────

_ELEVENLABS_VOICES = {
    "en": "21m00Tcm4TlvDq8ikWAM",   # Rachel — clear Indian-accented English
}

_AZURE_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "en": "en-IN-NeerjaNeural",      # fallback English
}

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
AZURE_TTS_URL = "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"


class TTSStreamer:
    """
    Language-aware TTS that streams audio bytes.
    Barge-in: call cancel() to abort current stream.
    """

    def __init__(
        self,
        elevenlabs_api_key: str,
        azure_subscription_key: str,
        azure_region: str = "centralindia",
    ):
        self.el_key = elevenlabs_api_key
        self.az_key = azure_subscription_key
        self.az_region = azure_region
        self._cancel_event: Optional[asyncio.Event] = None

    def cancel(self):
        """Call this to interrupt mid-stream (barge-in)."""
        if self._cancel_event:
            self._cancel_event.set()

    async def stream(
        self,
        text: str,
        language: str = "en",
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio bytes for the given text.
        First chunk arrives in ~80ms on ElevenLabs (English).
        """
        self._cancel_event = asyncio.Event()

        if language == "en":
            async for chunk in self._stream_elevenlabs(text, self._cancel_event):
                yield chunk
        else:
            async for chunk in self._stream_azure(text, language, self._cancel_event):
                yield chunk

    async def _stream_elevenlabs(
        self,
        text: str,
        cancel: asyncio.Event,
    ) -> AsyncGenerator[bytes, None]:
        voice_id = _ELEVENLABS_VOICES["en"]
        url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)

        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
            "output_format": "pcm_16000",  # 16kHz PCM for lowest latency
        }

        headers = {
            "xi-api-key": self.el_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        t_start = time.monotonic()
        first_chunk = True

        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if cancel.is_set():
                        logger.info("TTS stream cancelled (barge-in)")
                        return
                    if first_chunk:
                        logger.info(
                            "TTS first chunk in %.1fms", (time.monotonic() - t_start) * 1000
                        )
                        first_chunk = False
                    yield chunk

    async def _stream_azure(
        self,
        text: str,
        language: str,
        cancel: asyncio.Event,
    ) -> AsyncGenerator[bytes, None]:
        voice = _AZURE_VOICES.get(language, _AZURE_VOICES["en"])
        url = AZURE_TTS_URL.format(region=self.az_region)

        ssml = f"""<speak version='1.0' xml:lang='{language}'>
            <voice name='{voice}'>{text}</voice>
        </speak>"""

        headers = {
            "Ocp-Apim-Subscription-Key": self.az_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "raw-16khz-16bit-mono-pcm",
        }

        t_start = time.monotonic()
        first_chunk = True

        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", url, content=ssml.encode(), headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if cancel.is_set():
                        logger.info("TTS stream cancelled (barge-in)")
                        return
                    if first_chunk:
                        logger.info(
                            "TTS Azure first chunk in %.1fms", (time.monotonic() - t_start) * 1000
                        )
                        first_chunk = False
                    yield chunk
