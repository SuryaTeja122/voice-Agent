"""
agent/tools.py

All agent tools with Anthropic tool schemas.
Tools are genuinely implemented — no hardcoded responses.
"""

import logging
from typing import Any
from datetime import datetime, timedelta

from services.scheduling import SchedulingService
from services.patient import PatientService

logger = logging.getLogger(__name__)

# ─── Tool schemas (passed to Claude API) ──────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "check_availability",
        "description": (
            "Check available appointment slots for a doctor on a given date or date range. "
            "Use this before booking to show options to the patient."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doctor_name": {"type": "string", "description": "Doctor's name (partial match allowed)"},
                "specialty": {"type": "string", "description": "Medical specialty if doctor name unknown"},
                "date": {"type": "string", "description": "ISO date YYYY-MM-DD or 'today'/'tomorrow'/'this week'"},
                "preferred_time": {"type": "string", "description": "morning | afternoon | evening | any"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "book_appointment",
        "description": (
            "Book a confirmed appointment. Only call this after the patient has explicitly confirmed "
            "the doctor, date, and time. Handles double-booking prevention automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "doctor_id": {"type": "string", "description": "Use doctor_id from check_availability result"},
                "slot_time": {"type": "string", "description": "ISO datetime YYYY-MM-DDTHH:MM:SS"},
                "reason": {"type": "string", "description": "Brief reason for visit"},
            },
            "required": ["patient_id", "doctor_id", "slot_time"],
        },
    },
    {
        "name": "reschedule_appointment",
        "description": "Reschedule an existing appointment to a new slot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string"},
                "new_slot_time": {"type": "string", "description": "ISO datetime for new slot"},
                "reason": {"type": "string"},
            },
            "required": ["appointment_id", "new_slot_time"],
        },
    },
    {
        "name": "cancel_appointment",
        "description": "Cancel an existing appointment. Always confirm with patient before calling.",
        "input_schema": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["appointment_id"],
        },
    },
    {
        "name": "get_patient_appointments",
        "description": "Retrieve a patient's upcoming and recent past appointments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "include_past": {"type": "boolean", "default": False},
            },
            "required": ["patient_id"],
        },
    },
    {
        "name": "get_patient_history",
        "description": "Get patient profile, preferences, and past visit summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
            },
            "required": ["patient_id"],
        },
    },
    {
        "name": "log_rejection",
        "description": (
            "Log when a patient declines an appointment reminder or follow-up. "
            "Records reason for future campaign suppression."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "campaign_id": {"type": "string"},
                "reason": {"type": "string", "description": "Patient's stated reason or 'no_reason'"},
            },
            "required": ["patient_id", "campaign_id"],
        },
    },
    {
        "name": "find_doctor",
        "description": "Search for doctors by name, specialty, or availability.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Doctor name or specialty"},
                "language_spoken": {"type": "string", "description": "en | hi | ta — prefer doctors who speak this language"},
            },
            "required": ["query"],
        },
    },
]

# ─── Tool dispatcher ──────────────────────────────────────────────────────────

# These are injected at startup; set via init_tools()
_scheduling: SchedulingService = None
_patient: PatientService = None


def init_tools(scheduling: SchedulingService, patient: PatientService):
    global _scheduling, _patient
    _scheduling = scheduling
    _patient = patient


async def dispatch_tool(name: str, args: dict[str, Any]) -> dict:
    """Routes tool calls to implementations. All errors are returned as structured dicts."""
    try:
        match name:
            case "check_availability":
                return await _check_availability(args)
            case "book_appointment":
                return await _book_appointment(args)
            case "reschedule_appointment":
                return await _reschedule_appointment(args)
            case "cancel_appointment":
                return await _cancel_appointment(args)
            case "get_patient_appointments":
                return await _get_patient_appointments(args)
            case "get_patient_history":
                return await _get_patient_history(args)
            case "log_rejection":
                return await _log_rejection(args)
            case "find_doctor":
                return await _find_doctor(args)
            case _:
                return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        logger.exception("Tool %s failed with args %s", name, args)
        return {"error": str(e), "tool": name}


# ─── Tool implementations ─────────────────────────────────────────────────────

async def _check_availability(args: dict) -> dict:
    date_str = args.get("date", "today")
    doctor_name = args.get("doctor_name")
    specialty = args.get("specialty")
    preferred_time = args.get("preferred_time", "any")

    slots = await _scheduling.get_available_slots(
        date_str=date_str,
        doctor_name=doctor_name,
        specialty=specialty,
        preferred_time=preferred_time,
    )
    return {
        "available_slots": slots,
        "count": len(slots),
        "date_queried": date_str,
    }


async def _book_appointment(args: dict) -> dict:
    result = await _scheduling.book_slot(
        patient_id=args["patient_id"],
        doctor_id=args["doctor_id"],
        slot_time=args["slot_time"],
        reason=args.get("reason", ""),
    )
    if result.success:
        return {
            "success": True,
            "appointment_id": result.appointment_id,
            "confirmation_code": result.confirmation_code,
            "slot_time": args["slot_time"],
            "doctor_id": args["doctor_id"],
        }
    else:
        return {
            "success": False,
            "reason": result.conflict_reason,
            "alternatives": result.alternatives,
        }


async def _reschedule_appointment(args: dict) -> dict:
    result = await _scheduling.reschedule(
        appointment_id=args["appointment_id"],
        new_slot_time=args["new_slot_time"],
        reason=args.get("reason", ""),
    )
    return {
        "success": result.success,
        "appointment_id": args["appointment_id"],
        "new_slot_time": args["new_slot_time"],
        "conflict": result.conflict_reason if not result.success else None,
        "alternatives": result.alternatives if not result.success else [],
    }


async def _cancel_appointment(args: dict) -> dict:
    success = await _scheduling.cancel(
        appointment_id=args["appointment_id"],
        reason=args.get("reason", ""),
    )
    return {"success": success, "appointment_id": args["appointment_id"]}


async def _get_patient_appointments(args: dict) -> dict:
    appointments = await _patient.get_appointments(
        patient_id=args["patient_id"],
        include_past=args.get("include_past", False),
    )
    return {"appointments": appointments}


async def _get_patient_history(args: dict) -> dict:
    profile = await _patient.get_profile(args["patient_id"])
    if not profile:
        return {"error": "Patient not found"}
    return {
        "patient_id": profile.id,
        "name": profile.name,
        "language_preference": profile.language_preference,
        "last_visit": profile.last_visit_summary,
        "preferred_doctors": profile.preferred_doctors,
        "preferred_time": profile.preferred_time_of_day,
    }


async def _log_rejection(args: dict) -> dict:
    await _patient.log_campaign_rejection(
        patient_id=args["patient_id"],
        campaign_id=args["campaign_id"],
        reason=args.get("reason", "no_reason"),
    )
    return {"logged": True}


async def _find_doctor(args: dict) -> dict:
    doctors = await _scheduling.find_doctors(
        query=args["query"],
        language_spoken=args.get("language_spoken"),
    )
    return {"doctors": doctors, "count": len(doctors)}
