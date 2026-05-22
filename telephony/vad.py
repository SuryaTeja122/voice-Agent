"""
telephony/vad.py

Voice Activity Detection using Silero VAD.
Handles:
- End-of-speech detection → triggers STT finalization
- Barge-in: speech detected while TTS is playing → cancel TTS, restart STT
"""

import asyncio
import logging
import time
from collections import deque
from typing import Callable, Optional, AsyncGenerator

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30          # Silero works on 30ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples
SPEECH_THRESHOLD = 0.5
SILENCE_FRAMES_TO_END = 10      # 300ms of silence = end of speech
PRE_ROLL_FRAMES = 5             # 150ms pre-roll to capture speech onset


class SileroVAD:
    """
    Wraps Silero VAD model (torch).
    Falls back to energy-based VAD if torch not available.
    """

    def __init__(self):
        self._model = None
        self._torch = None
        self._load()

    def _load(self):
        try:
            import torch
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,         # ONNX for faster inference
            )
            self._model = model
            self._torch = torch
            logger.info("Silero VAD loaded (ONNX)")
        except Exception as e:
            logger.warning("Silero VAD unavailable, using energy VAD: %s", e)

    def is_speech(self, frame: bytes) -> float:
        """Returns speech probability [0, 1] for a 30ms PCM16 frame."""
        if self._model and self._torch:
            audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = self._torch.from_numpy(audio).unsqueeze(0)
            with self._torch.no_grad():
                prob = self._model(tensor, SAMPLE_RATE).item()
            return prob
        else:
            # Energy-based fallback
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            energy = np.sqrt(np.mean(samples ** 2))
            return min(1.0, energy / 2000.0)


class VADPipeline:
    """
    Full VAD pipeline:
    1. Buffers incoming audio frames
    2. Detects speech onset → notifies listener
    3. Detects end-of-speech → delivers complete utterance audio
    4. Handles barge-in during TTS playback
    """

    def __init__(
        self,
        vad: SileroVAD,
        on_speech_start: Optional[Callable] = None,
        on_barge_in: Optional[Callable] = None,
    ):
        self.vad = vad
        self.on_speech_start = on_speech_start  # called when speech begins
        self.on_barge_in = on_barge_in          # called when speech during TTS
        self._is_tts_playing = False
        self._pre_roll: deque = deque(maxlen=PRE_ROLL_FRAMES)

    def set_tts_playing(self, playing: bool):
        self._is_tts_playing = playing

    async def process_stream(
        self,
        audio_source: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[bytes, None]:
        """
        Processes raw audio stream.
        Yields complete utterance audio chunks (pre-roll + speech + post-roll).
        """
        speech_frames = []
        silence_count = 0
        in_speech = False
        speech_started_at = None

        async for raw_chunk in audio_source:
            # Process in 30ms frames
            for i in range(0, len(raw_chunk) - FRAME_SAMPLES * 2, FRAME_SAMPLES * 2):
                frame = raw_chunk[i : i + FRAME_SAMPLES * 2]
                if len(frame) < FRAME_SAMPLES * 2:
                    continue

                prob = self.vad.is_speech(frame)
                is_speech = prob >= SPEECH_THRESHOLD

                # ── Barge-in detection ────────────────────────────────────
                if is_speech and self._is_tts_playing:
                    logger.info("Barge-in detected (prob=%.2f)", prob)
                    if self.on_barge_in:
                        self.on_barge_in()
                    # Fall through to normal speech handling

                # ── Speech onset ──────────────────────────────────────────
                if is_speech and not in_speech:
                    in_speech = True
                    silence_count = 0
                    speech_started_at = time.monotonic()
                    if self.on_speech_start:
                        self.on_speech_start()
                    # Include pre-roll frames
                    speech_frames = list(self._pre_roll) + [frame]
                    logger.debug("Speech onset")

                elif is_speech and in_speech:
                    speech_frames.append(frame)
                    silence_count = 0

                elif not is_speech and in_speech:
                    silence_count += 1
                    speech_frames.append(frame)  # include trailing silence

                    if silence_count >= SILENCE_FRAMES_TO_END:
                        # End of speech — yield complete utterance
                        duration_ms = (time.monotonic() - speech_started_at) * 1000
                        logger.info("Speech ended after %.0fms", duration_ms)
                        utterance = b"".join(speech_frames[:-SILENCE_FRAMES_TO_END])
                        yield utterance
                        speech_frames = []
                        in_speech = False
                        silence_count = 0

                else:
                    # Silence while not in speech — maintain pre-roll
                    self._pre_roll.append(frame)

        # Flush any remaining speech at stream end
        if speech_frames and in_speech:
            yield b"".join(speech_frames)
