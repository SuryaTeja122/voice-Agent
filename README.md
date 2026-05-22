# Real-Time Multilingual Voice AI Agent — Clinical Appointment Booking

> Target: **<450ms** end-to-end latency (speech-end → first audio byte)  
> Languages: **English · Hindi · Tamil**  
> Stack: **Python 3.11 · FastAPI · TypeScript (admin UI)**

---

## Quick Start

```bash
# 1. Clone and set up
git clone <repo>
cd voice-agent
cp .env.example .env           # fill in API keys

# 2. Start infrastructure
docker compose up -d           # Redis, PostgreSQL, pgvector

# 3. Run migrations
alembic upgrade head

# 4. Start the agent server
uvicorn api.main:app --reload --port 8000

# 5. Start the Celery worker (outbound campaigns)
celery -A workers.celery_app worker --loglevel=info -Q campaigns,reminders
```

### Environment Variables

```
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY for GPT-4o
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql+asyncpg://...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=...
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Voice I/O Layer                          │
│  Inbound WebRTC ──► STT (Deepgram streaming)                   │
│  Outbound Twilio ◄── TTS (ElevenLabs chunked)                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ text + language tag
┌─────────────────────────▼───────────────────────────────────────┐
│                      Agent Core                                  │
│  VAD / barge-in ──► LLM Orchestrator ──► Tool Router           │
│                    (Claude 3.5 Sonnet)                          │
│                    + Session State Manager                      │
└──────────┬────────────────────┬────────────────────┬────────────┘
           │                    │                    │
    ┌──────▼──────┐    ┌────────▼───────┐   ┌───────▼────────┐
    │  Scheduling │    │ Patient Service│   │ Campaign Sched.│
    │  Engine     │    │                │   │ (Celery+Redis) │
    └──────┬──────┘    └────────┬───────┘   └───────┬────────┘
           │                    │                    │
┌──────────▼────────────────────▼────────────────────▼────────────┐
│                     Memory & Storage                             │
│  Redis (session TTL)  │  PostgreSQL (long-term)  │  pgvector    │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Role |
|-----------|------|
| `telephony/stt.py` | Deepgram WebSocket streaming, returns word-by-word transcripts with language detection |
| `telephony/tts.py` | ElevenLabs streaming TTS, first-chunk dispatch <80ms, Hindi/Tamil voice models |
| `telephony/vad.py` | Silero VAD for end-of-speech detection; barge-in interrupt via buffer flush |
| `agent/orchestrator.py` | LLM call with tool schemas, conversation loop, reasoning trace emission |
| `agent/tools.py` | All callable tools: `book_appointment`, `reschedule`, `cancel`, `check_availability`, `get_patient_history`, `log_rejection` |
| `agent/session.py` | Per-call state: current intent, pending confirmation, language, partial data |
| `services/scheduling.py` | Slot locking, conflict detection, alternative suggestions, double-booking prevention |
| `services/patient.py` | Patient CRUD, history retrieval, language preference persistence |
| `memory/session_store.py` | Redis-backed session with configurable TTL (default 2h) |
| `memory/long_term.py` | PostgreSQL writes; pgvector embedding + retrieval for past interaction context |
| `workers/campaign.py` | Celery tasks: schedule outbound calls, retry logic, rejection logging |
| `api/main.py` | FastAPI: WebSocket endpoint for calls, REST for admin/health |

---

## Latency Breakdown

| Stage | Budget | Implementation |
|-------|--------|----------------|
| Network + WebRTC ingress | ~30ms | edge PoP termination |
| VAD end-of-speech detect | ~20ms | Silero, 30ms frame size |
| STT (Deepgram streaming) | ~80ms | interim results; final on end-of-speech |
| Context fetch (Redis + PG) | ~0ms | **parallel** with STT final wait |
| LLM first token | ~150ms | Claude 3.5 Sonnet w/ streaming |
| Tool execution (if needed) | ~40ms | DB query, atomic slot lock |
| TTS first audio chunk | ~80ms | ElevenLabs streaming, text pre-buffered |
| Audio egress to client | ~30ms | WebRTC |
| **Total** | **~430ms** | under 450ms target |

**Key optimisation:** context fetch (Redis session + pgvector retrieval) fires in parallel with the last 100ms of STT finalization, eliminating it from the critical path.

All stages are instrumented with OpenTelemetry spans. Each call logs a `latency_report.jsonl` entry:

```json
{
  "call_id": "c_abc123",
  "stt_ms": 78,
  "context_fetch_ms": 22,
  "llm_first_token_ms": 148,
  "tool_exec_ms": 38,
  "tts_first_chunk_ms": 77,
  "total_ms": 421,
  "language": "hi",
  "timestamp": "2025-08-01T10:32:11Z"
}
```

---

## Memory Design

### Two-Level Architecture

**Level 1 — Session memory (Redis)**
- Key: `session:{call_id}`  
- Value: serialised `SessionState` (current intent, slots collected, pending confirmation, language, patient_id)  
- TTL: 2 hours (configurable)  
- Used: every LLM turn; injected as the first system message section

```python
@dataclass
class SessionState:
    call_id: str
    patient_id: Optional[str]
    language: str          # "en" | "hi" | "ta"
    intent: Optional[str]  # "book" | "reschedule" | "cancel" | None
    slots: dict            # {"doctor": "Dr. Meera", "date": "2025-08-05", ...}
    pending_confirm: Optional[dict]
    turn_count: int
    started_at: datetime
```

**Level 2 — Long-term memory (PostgreSQL + pgvector)**
- `patients` table: demographics, language preference, last interaction
- `appointments` table: full lifecycle with status enum
- `interaction_summaries` table: post-call summaries with embedding (text-embedding-3-small, 1536d)
- At call start: top-3 semantically similar past interactions retrieved via cosine similarity and injected into system prompt under `<patient_context>`

**Prompt integration pattern:**

```
System:
  You are a clinical appointment assistant for 2Care Health.
  
  <patient_context>
  Patient: Ravi Kumar | Language pref: Hindi
  Last visit: 2025-06-12, Dr. Anand, Cardiology — rescheduled due to travel
  Past preference: morning slots, weekdays
  </patient_context>
  
  <session_state>
  Current intent: book | Slots: {doctor: "Dr. Anand"} | Turn: 2
  </session_state>
  
  Respond only in the patient's detected language (hi). ...
```

### Why not a pure vector DB?

Redis gives sub-millisecond session reads on the hot path. PostgreSQL gives ACID guarantees for slot locking (critical for double-booking prevention). pgvector avoids a separate service while still enabling semantic retrieval. The combination covers all three access patterns at minimal operational overhead.

---

## Scheduling & Conflict Logic

```python
# Atomic slot locking — prevents double-bookings under concurrent calls
async def book_slot(doctor_id, slot_dt, patient_id, conn) -> BookingResult:
    async with conn.transaction():
        # Advisory lock on (doctor_id, slot_dt) — blocks concurrent bookings
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext($1 || $2::text))",
            doctor_id, slot_dt.isoformat()
        )
        existing = await conn.fetchrow(
            "SELECT id FROM appointments WHERE doctor_id=$1 AND slot_time=$2 AND status!='cancelled'",
            doctor_id, slot_dt
        )
        if existing:
            alternatives = await suggest_alternatives(doctor_id, slot_dt, conn)
            return BookingResult(success=False, alternatives=alternatives)
        
        appt_id = await conn.fetchval(
            "INSERT INTO appointments (...) VALUES (...) RETURNING id",
            ...
        )
        return BookingResult(success=True, appointment_id=appt_id)
```

**Conflict scenarios handled:**
- Double-booking: atomic lock + alternative suggestion
- Past-time selection: validated before LLM tool call
- Doctor unavailable: check against `doctor_availability` table; offer same-specialty alternatives
- Mid-conversation change of mind: session state reset, previous tentative lock released
- Unclear request: clarification turn triggered by LLM, not hardcoded

---

## Multilingual Support

| Language | STT Model | TTS Voice | Detection |
|----------|-----------|-----------|-----------|
| English | `deepgram:nova-2` | ElevenLabs `Rachel` | fastText |
| Hindi | `deepgram:nova-2-general` (hi) | ElevenLabs `Hindi Female` / Azure Neural | fastText |
| Tamil | `deepgram:nova-2-general` (ta) | Azure Neural `ta-IN-PallaviNeural` | fastText |

Language detection runs on the first 2 turns and persists to the patient record. Mid-call language switches are detected per-turn and handled gracefully (the agent acknowledges and continues in the new language).

---

## Outbound Campaign Mode

```
Celery beat ──► campaign task ──► Twilio outbound call
                                       │
                                  Agent joins call
                                       │
                            patient response handled:
                            - books/reschedules → DB write
                            - polite rejection  → log_rejection tool
                            - no answer         → retry queue (max 3)
```

Campaign job schema:

```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=3600)
def initiate_reminder_call(self, patient_id: str, appointment_id: str):
    ...
```

---

## Bonus Features Implemented

- **Barge-in handling**: Silero VAD monitors audio while TTS plays; detected speech flushes the TTS buffer and re-enters the STT pipeline
- **Redis-backed memory with TTL**: session store with 2h TTL, campaign job state with 24h TTL
- **Background job queues**: Celery with Redis broker, separate queues for `campaigns` and `reminders`
- **Horizontal scalability**: stateless agent workers behind a load balancer; session state in Redis (shared); DB connection pooling via asyncpg

---

## Tradeoffs & Known Limitations

| Decision | Tradeoff |
|----------|----------|
| Deepgram over Whisper | Lower latency (~80ms vs ~300ms) but higher per-minute cost |
| Claude 3.5 Sonnet | Best tool-calling accuracy; slightly higher latency than GPT-4o-mini |
| PostgreSQL advisory locks | Simpler than Redis SETNX but ties slot locking to DB connection |
| fastText for language detection | Fast but may misdetect code-mixed Hinglish on first turn |
| pgvector in-process | No separate infra, but doesn't scale to millions of embeddings |
| ElevenLabs for TTS | Best voice quality for English; Hindi/Tamil quality is acceptable but not native-grade |

**Known limitations:**
- Tamil ASR accuracy degrades with heavy accents; fallback to spelling confirmation turn recommended
- Concurrent campaign calls > 50 require Twilio capacity planning
- pgvector HNSW index rebuild needed if interaction_summaries > 500k rows
- No HIPAA compliance layer yet (PHI encryption at rest + audit log required for production)

---

## Project Structure

```
voice-agent/
├── api/
│   ├── main.py              # FastAPI app, WebSocket /call endpoint
│   ├── routes/
│   │   ├── calls.py
│   │   ├── campaigns.py
│   │   └── health.py
├── agent/
│   ├── orchestrator.py      # LLM loop, tool dispatch, trace emission
│   ├── tools.py             # All tool implementations
│   ├── session.py           # SessionState dataclass + helpers
│   └── prompts.py           # System prompt builder
├── telephony/
│   ├── stt.py               # Deepgram WebSocket client
│   ├── tts.py               # ElevenLabs / Azure TTS streaming
│   └── vad.py               # Silero VAD + barge-in
├── services/
│   ├── scheduling.py        # Slot locking, conflict, alternatives
│   └── patient.py           # Patient CRUD + language preference
├── memory/
│   ├── session_store.py     # Redis session (get/set/delete)
│   └── long_term.py         # PostgreSQL + pgvector reads/writes
├── workers/
│   ├── celery_app.py        # Celery + Redis broker config
│   └── campaign.py          # Outbound call tasks
├── db/
│   ├── models.py            # SQLAlchemy models
│   └── migrations/          # Alembic
├── tests/
│   ├── test_scheduling.py
│   ├── test_agent.py
│   └── test_latency.py
├── docs/
│   └── architecture.png
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```
