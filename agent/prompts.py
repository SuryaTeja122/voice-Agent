"""
agent/prompts.py

Builds the system prompt for each LLM turn.
Injects: language instruction, patient context, session state, tool guidance.
"""

from typing import Optional
from agent.session import SessionState

_LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
}

_LANGUAGE_INSTRUCTIONS = {
    "en": "Respond in clear, friendly English. Use simple medical terminology.",
    "hi": (
        "हिंदी में जवाब दें। स्पष्ट और सरल भाषा का उपयोग करें। "
        "Medical terms के लिए English words acceptable हैं।"
    ),
    "ta": (
        "தமிழில் பதில் அளிக்கவும். எளிய மொழியில் பேசவும். "
        "Medical terms-க்கு English words பயன்படுத்தலாம்."
    ),
}

_BASE_SYSTEM = """\
You are a friendly, professional clinical appointment assistant for 2Care Health, a digital healthcare platform in India.

Your primary responsibilities:
- Book, reschedule, and cancel clinical appointments through natural conversation
- Detect and respond in the patient's language (English, Hindi, or Tamil)
- Handle conflicting slots gracefully and offer alternatives
- Confirm all bookings/cancellations before executing them

Important guidelines:
- ALWAYS confirm appointment details with the patient before calling book_appointment or cancel_appointment
- If a slot is unavailable, immediately offer 2–3 alternatives from check_availability
- If the patient changes their mind mid-conversation, acknowledge it and restart from their new intent
- For unclear requests, ask ONE clarifying question — do not ask multiple questions at once
- Keep responses concise — this is a voice call, not a chat interface
- Never fabricate doctor names, slots, or appointment IDs — always use tool results
- Reasoning: think through the patient's intent before deciding which tool to call

{language_instruction}
"""

_SESSION_STATE_SECTION = """\
<session_state>
Intent: {intent}
Collected slots: {slots}
Turn number: {turn_count}
{pending_section}
</session_state>"""

_PATIENT_CONTEXT_SECTION = """\
<patient_context>
{context}
</patient_context>"""

_OUTBOUND_ADDITION = """\

This is an OUTBOUND call initiated for a campaign. You called the patient — they did not call you.
Be brief and respectful of their time. Introduce yourself in the first turn.
If they decline or ask to be called later, use log_rejection and end the call politely.
Campaign ID: {campaign_id}
"""


def build_system_prompt(
    language: str,
    patient_context: Optional[str],
    session: SessionState,
) -> str:
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["en"])
    base = _BASE_SYSTEM.format(language_instruction=lang_instruction)

    # Patient context section (from pgvector RAG)
    if patient_context:
        base += "\n\n" + _PATIENT_CONTEXT_SECTION.format(context=patient_context)

    # Session state section
    pending_section = ""
    if session.pending_confirm:
        pending_section = f"Pending confirmation: {session.pending_confirm}"

    base += "\n\n" + _SESSION_STATE_SECTION.format(
        intent=session.intent or "unknown",
        slots=session.slots or "none collected",
        turn_count=session.turn_count,
        pending_section=pending_section,
    )

    # Outbound campaign instruction
    if session.is_outbound:
        base += _OUTBOUND_ADDITION.format(campaign_id=session.campaign_id)

    return base
