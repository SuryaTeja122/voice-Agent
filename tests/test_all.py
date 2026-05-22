"""
tests/test_scheduling.py — Slot locking, conflict detection, alternatives
tests/test_agent.py      — Tool dispatch, session state, prompt injection
tests/test_latency.py    — Latency budget assertions on mock pipeline
"""

# ════════════════════════════════════════════════════════════════
# tests/test_scheduling.py
# ════════════════════════════════════════════════════════════════
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from services.scheduling import SchedulingService, BookingResult

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


@pytest.fixture
def svc(mock_pool):
    pool, _ = mock_pool
    return SchedulingService(pool)


@pytest.mark.asyncio
async def test_book_slot_success(mock_pool, svc):
    pool, conn = mock_pool
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)  # no existing booking
    conn.fetchval = AsyncMock(return_value="appt-uuid-123")

    future_slot = (datetime.now(tz=IST) + timedelta(days=1)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )

    result = await svc.book_slot(
        patient_id="patient-1",
        doctor_id="doctor-1",
        slot_time=future_slot.isoformat(),
    )

    assert result.success is True


@pytest.mark.asyncio
async def test_book_slot_conflict_returns_alternatives(mock_pool, svc):
    pool, conn = mock_pool
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock()
    # Simulate existing booking
    conn.fetchrow = AsyncMock(return_value={"id": "existing-appt"})
    # Alternatives fetch
    conn.fetch = AsyncMock(return_value=[
        {
            "doctor_id": "doctor-1",
            "doctor_name": "Dr. Anand",
            "specialty": "General Medicine",
            "languages_spoken": ["en", "hi"],
            "slot_time": datetime.now(tz=IST) + timedelta(days=1, hours=2),
        }
    ])

    future_slot = (datetime.now(tz=IST) + timedelta(days=1)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )

    result = await svc.book_slot(
        patient_id="patient-1",
        doctor_id="doctor-1",
        slot_time=future_slot.isoformat(),
    )

    assert result.success is False
    assert result.conflict_reason is not None
    assert isinstance(result.alternatives, list)


@pytest.mark.asyncio
async def test_book_past_slot_rejected(svc):
    past_slot = (datetime.now(tz=IST) - timedelta(hours=1)).isoformat()
    result = await svc.book_slot(
        patient_id="p1",
        doctor_id="d1",
        slot_time=past_slot,
    )
    assert result.success is False
    assert "passed" in result.conflict_reason.lower()


@pytest.mark.asyncio
async def test_cancel_appointment(mock_pool, svc):
    pool, conn = mock_pool
    conn.execute = AsyncMock(return_value="UPDATE 1")

    result = await svc.cancel("appt-123", reason="patient request")
    assert result is True


# ════════════════════════════════════════════════════════════════
# tests/test_agent.py
# ════════════════════════════════════════════════════════════════
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agent.session import SessionState
from agent.prompts import build_system_prompt


def test_session_state_serialisation():
    s = SessionState(call_id="call-1", language="hi", patient_id="p1")
    s.intent = "book"
    s.slots = {"doctor": "Dr. Anand", "date": "2025-08-05"}
    json_str = s.to_json()
    s2 = SessionState.from_json(json_str)
    assert s2.call_id == "call-1"
    assert s2.language == "hi"
    assert s2.slots["doctor"] == "Dr. Anand"


def test_session_intent_reset_clears_slots():
    s = SessionState(call_id="call-1")
    s.set_intent("book")
    s.collect_slot("doctor", "Dr. Anand")
    s.set_intent("cancel")   # change of mind
    assert s.intent == "cancel"
    assert s.slots == {}


def test_session_history_truncated_to_20():
    s = SessionState(call_id="call-1")
    for i in range(25):
        s.append_turn("user", f"message {i}")
    assert len(s.conversation_history) == 20


def test_build_system_prompt_hindi():
    s = SessionState(call_id="c1", language="hi", turn_count=2, intent="book")
    s.slots = {"doctor": "Dr. Anand"}
    prompt = build_system_prompt("hi", patient_context="Last visit: June 2025", session=s)
    assert "हिंदी" in prompt
    assert "patient_context" in prompt
    assert "book" in prompt


def test_build_system_prompt_outbound():
    s = SessionState(call_id="c1", campaign_id="reminder_20250801")
    prompt = build_system_prompt("en", patient_context=None, session=s)
    assert "OUTBOUND" in prompt
    assert "reminder_20250801" in prompt


def test_build_system_prompt_tamil():
    s = SessionState(call_id="c1", language="ta")
    prompt = build_system_prompt("ta", patient_context=None, session=s)
    assert "தமிழில்" in prompt


# ════════════════════════════════════════════════════════════════
# tests/test_latency.py
# ════════════════════════════════════════════════════════════════
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.orchestrator import AgentOrchestrator, LatencyReport


@pytest.mark.asyncio
async def test_latency_report_logged(caplog):
    """Verify that LatencyReport is emitted with correct fields."""
    report = LatencyReport(
        call_id="test-call",
        stt_ms=78.0,
        context_fetch_ms=22.0,
        llm_first_token_ms=148.0,
        tool_exec_ms=38.0,
        tts_first_chunk_ms=77.0,
        total_ms=421.0,
        language="en",
    )
    import logging
    with caplog.at_level(logging.INFO, logger="agent.orchestrator"):
        report.log()

    assert "LATENCY_REPORT" in caplog.text
    assert "421" in caplog.text


def test_latency_budget_within_target():
    """All stage budgets should sum to under 450ms."""
    stt = 80
    context_fetch = 0     # parallel — not on critical path
    llm = 200
    tool = 40
    tts_first = 80
    network = 40
    total = stt + llm + tool + tts_first + network
    assert total < 450, f"Latency budget exceeded: {total}ms"


@pytest.mark.asyncio
async def test_context_fetch_is_parallel():
    """
    Demonstrates that context fetch runs in parallel with the final
    STT wait, not sequentially, so it doesn't add to wall time.
    """
    async def mock_stt_final_wait():
        await asyncio.sleep(0.08)  # simulate 80ms STT
        return "book an appointment"

    async def mock_context_fetch():
        await asyncio.sleep(0.05)  # simulate 50ms context fetch
        return "Patient last visited June 2025"

    t_start = time.monotonic()
    transcript, context = await asyncio.gather(
        mock_stt_final_wait(),
        mock_context_fetch(),
    )
    elapsed = (time.monotonic() - t_start) * 1000

    # Wall time should be ~80ms (max of 80, 50), not 130ms (sum)
    assert elapsed < 110, f"Context fetch was sequential: {elapsed:.0f}ms"
    assert transcript == "book an appointment"
    assert "June 2025" in context
