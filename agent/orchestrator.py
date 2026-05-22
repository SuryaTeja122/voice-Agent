"""
agent/orchestrator.py

Core agent loop: receives transcribed text, fetches context, calls LLM,
dispatches tools, streams TTS response. All latency checkpoints are logged.
"""

import asyncio
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import AsyncGenerator, Optional

import anthropic

from agent.session import SessionState
from agent.tools import TOOL_SCHEMAS, dispatch_tool
from agent.prompts import build_system_prompt
from memory.session_store import SessionStore
from memory.long_term import LongTermMemory
from telephony.tts import TTSStreamer

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"


@dataclass
class LatencyReport:
    call_id: str
    stt_ms: float
    context_fetch_ms: float
    llm_first_token_ms: float
    tool_exec_ms: float
    tts_first_chunk_ms: float
    total_ms: float
    language: str

    def log(self):
        logger.info("LATENCY_REPORT %s", json.dumps(asdict(self)))


class AgentOrchestrator:
    def __init__(
        self,
        session_store: SessionStore,
        long_term_memory: LongTermMemory,
        tts_streamer: TTSStreamer,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        self.session_store = session_store
        self.ltm = long_term_memory
        self.tts = tts_streamer
        self.client = anthropic_client

    async def handle_turn(
        self,
        call_id: str,
        transcript: str,
        language: str,
        stt_duration_ms: float,
    ) -> AsyncGenerator[bytes, None]:
        """
        Full pipeline for one conversation turn.
        Yields audio bytes as they stream from TTS.
        """
        t_start = time.monotonic()

        # 1. Fetch session + long-term context in parallel with STT final wait
        t_ctx_start = time.monotonic()
        session, patient_context = await asyncio.gather(
            self.session_store.get(call_id),
            self._fetch_patient_context(call_id),
        )
        context_fetch_ms = (time.monotonic() - t_ctx_start) * 1000

        if session is None:
            session = SessionState(call_id=call_id, language=language)

        # Update language from this turn
        session.language = language
        session.turn_count += 1

        # 2. Build message history
        messages = session.to_messages()
        messages.append({"role": "user", "content": transcript})

        system_prompt = build_system_prompt(language, patient_context, session)

        # 3. LLM call with streaming + tool use
        t_llm_start = time.monotonic()
        llm_first_token_ms = 0.0
        tool_exec_ms = 0.0
        response_text = ""

        response_text, tool_exec_ms, llm_first_token_ms = await self._llm_with_tools(
            system_prompt, messages, t_llm_start
        )

        # 4. Update session with assistant reply
        messages.append({"role": "assistant", "content": response_text})
        session.update_from_response(response_text)
        await self.session_store.set(call_id, session)

        # 5. Stream TTS and yield audio chunks
        t_tts_start = time.monotonic()
        tts_first_chunk_ms = 0.0
        first_chunk = True

        async for audio_chunk in self.tts.stream(response_text, language):
            if first_chunk:
                tts_first_chunk_ms = (time.monotonic() - t_tts_start) * 1000
                first_chunk = False
            yield audio_chunk

        total_ms = (time.monotonic() - t_start) * 1000

        LatencyReport(
            call_id=call_id,
            stt_ms=stt_duration_ms,
            context_fetch_ms=context_fetch_ms,
            llm_first_token_ms=llm_first_token_ms,
            tool_exec_ms=tool_exec_ms,
            tts_first_chunk_ms=tts_first_chunk_ms,
            total_ms=total_ms,
            language=language,
        ).log()

    async def _llm_with_tools(
        self,
        system: str,
        messages: list,
        t_llm_start: float,
    ) -> tuple[str, float, float]:
        """
        Runs the LLM with tool-calling loop. Returns (response_text, tool_exec_ms, first_token_ms).
        Emits reasoning traces to logger.
        """
        first_token_ms = 0.0
        tool_exec_ms = 0.0
        full_response = ""
        current_messages = messages.copy()

        while True:
            stream = await self.client.messages.stream(
                model=ANTHROPIC_MODEL,
                max_tokens=512,
                system=system,
                messages=current_messages,
                tools=TOOL_SCHEMAS,
            )

            tool_calls = []
            response_content = []
            text_buffer = ""

            async with stream as s:
                async for event in s:
                    if first_token_ms == 0.0 and hasattr(event, "type"):
                        if event.type in ("content_block_delta", "content_block_start"):
                            first_token_ms = (time.monotonic() - t_llm_start) * 1000

                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            text_buffer += event.delta.text
                        elif hasattr(event.delta, "partial_json"):
                            # Accumulate tool input JSON
                            if tool_calls:
                                tool_calls[-1]["input_json"] = tool_calls[-1].get("input_json", "") + event.delta.partial_json

                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            tool_calls.append({
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input_json": "",
                            })

                    elif event.type == "message_stop":
                        break

            # Flush text
            if text_buffer:
                full_response = text_buffer
                response_content.append({"type": "text", "text": text_buffer})

            if not tool_calls:
                break

            # Process tools
            t_tools_start = time.monotonic()
            tool_results = []

            for tc in tool_calls:
                try:
                    tool_input = json.loads(tc["input_json"] or "{}")
                except json.JSONDecodeError:
                    tool_input = {}

                logger.info(
                    "TOOL_CALL call_id=? name=%s input=%s",
                    tc["name"],
                    json.dumps(tool_input),
                )

                result = await dispatch_tool(tc["name"], tool_input)

                logger.info("TOOL_RESULT name=%s result=%s", tc["name"], json.dumps(result))

                response_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tool_input,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps(result),
                })

            tool_exec_ms += (time.monotonic() - t_tools_start) * 1000

            # Continue conversation with tool results
            current_messages = current_messages + [
                {"role": "assistant", "content": response_content},
                {"role": "user", "content": tool_results},
            ]

        return full_response, tool_exec_ms, first_token_ms

    async def _fetch_patient_context(self, call_id: str) -> Optional[str]:
        """
        Retrieves long-term patient context from pgvector.
        Runs in parallel with session fetch to eliminate latency.
        """
        try:
            patient_id = await self.session_store.get_patient_id(call_id)
            if not patient_id:
                return None
            return await self.ltm.get_context_summary(patient_id, top_k=3)
        except Exception as e:
            logger.warning("Failed to fetch patient context: %s", e)
            return None
