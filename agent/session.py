"""
agent/session.py

SessionState: per-call in-memory state, serialised to/from Redis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class SessionState:
    call_id: str
    language: str = "en"                          # detected language: en | hi | ta
    patient_id: Optional[str] = None
    intent: Optional[str] = None                  # book | reschedule | cancel | query
    slots: dict = field(default_factory=dict)      # collected slot values
    pending_confirm: Optional[dict] = None         # awaiting patient yes/no
    turn_count: int = 0
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    conversation_history: list = field(default_factory=list)  # [{"role":..,"content":..}]
    campaign_id: Optional[str] = None             # set for outbound campaign calls

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "SessionState":
        data = json.loads(raw)
        return cls(**data)

    # ── Conversation history helpers ─────────────────────────────────────────

    def to_messages(self) -> list[dict]:
        """Returns conversation history in Anthropic messages format."""
        return list(self.conversation_history)

    def append_turn(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        # Keep last 20 turns in session to avoid context bloat
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    # ── State update helpers ─────────────────────────────────────────────────

    def update_from_response(self, assistant_text: str):
        """Called after each LLM response to append to history."""
        self.append_turn("assistant", assistant_text)

    def set_intent(self, intent: str):
        if self.intent != intent:
            self.intent = intent
            self.slots = {}  # reset slots when intent changes

    def collect_slot(self, key: str, value):
        self.slots[key] = value

    def set_pending_confirm(self, confirmation_payload: dict):
        self.pending_confirm = confirmation_payload

    def clear_pending_confirm(self):
        self.pending_confirm = None

    def reset_intent(self):
        """Called when patient changes their mind mid-conversation."""
        self.intent = None
        self.slots = {}
        self.pending_confirm = None

    @property
    def is_outbound(self) -> bool:
        return self.campaign_id is not None

    def summary_for_logging(self) -> dict:
        return {
            "call_id": self.call_id,
            "language": self.language,
            "patient_id": self.patient_id,
            "intent": self.intent,
            "turn_count": self.turn_count,
            "is_outbound": self.is_outbound,
        }
