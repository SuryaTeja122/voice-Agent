"""
services/scheduling.py

Appointment scheduling with:
- Atomic slot locking via PostgreSQL advisory locks
- Conflict detection and alternative suggestion
- Doctor availability management
- Double-booking prevention under concurrent calls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Optional
from zoneinfo import ZoneInfo

import asyncpg

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
SLOT_DURATION_MINUTES = 30


@dataclass
class BookingResult:
    success: bool
    appointment_id: Optional[str] = None
    confirmation_code: Optional[str] = None
    conflict_reason: Optional[str] = None
    alternatives: list = field(default_factory=list)


@dataclass
class SlotInfo:
    doctor_id: str
    doctor_name: str
    specialty: str
    slot_time: datetime
    slot_time_display: str   # human-readable: "Tuesday 3 August, 10:30 AM"
    languages_spoken: list[str]


class SchedulingService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_available_slots(
        self,
        date_str: str,
        doctor_name: Optional[str] = None,
        specialty: Optional[str] = None,
        preferred_time: str = "any",
        limit: int = 5,
    ) -> list[dict]:
        target_date = self._parse_date(date_str)

        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    d.id AS doctor_id,
                    d.name AS doctor_name,
                    d.specialty,
                    d.languages_spoken,
                    s.slot_time
                FROM doctor_availability s
                JOIN doctors d ON s.doctor_id = d.id
                WHERE
                    DATE(s.slot_time AT TIME ZONE 'Asia/Kolkata') = $1
                    AND s.is_available = TRUE
                    AND s.slot_time > NOW()
                    AND NOT EXISTS (
                        SELECT 1 FROM appointments a
                        WHERE a.doctor_id = d.id
                          AND a.slot_time = s.slot_time
                          AND a.status NOT IN ('cancelled')
                    )
                    {doctor_filter}
                    {specialty_filter}
                    {time_filter}
                ORDER BY s.slot_time
                LIMIT $2
            """.format(
                doctor_filter="AND LOWER(d.name) LIKE LOWER($3)" if doctor_name else "",
                specialty_filter="AND LOWER(d.specialty) = LOWER($4)" if specialty else "",
                time_filter=self._time_filter(preferred_time),
            )

            args = [target_date, limit]
            if doctor_name:
                args.append(f"%{doctor_name}%")
            if specialty:
                args.append(specialty)

            rows = await conn.fetch(query, *args)

        return [
            {
                "doctor_id": r["doctor_id"],
                "doctor_name": r["doctor_name"],
                "specialty": r["specialty"],
                "slot_time": r["slot_time"].isoformat(),
                "slot_display": self._format_slot(r["slot_time"]),
                "languages_spoken": r["languages_spoken"],
            }
            for r in rows
        ]

    async def book_slot(
        self,
        patient_id: str,
        doctor_id: str,
        slot_time: str,
        reason: str = "",
    ) -> BookingResult:
        slot_dt = datetime.fromisoformat(slot_time)

        # Validate: cannot book in the past
        if slot_dt < datetime.now(tz=IST):
            return BookingResult(
                success=False,
                conflict_reason="That time has already passed. Please choose a future slot.",
            )

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Advisory lock keyed on (doctor_id, slot_time) — prevents race conditions
                lock_key = hash(f"{doctor_id}:{slot_dt.isoformat()}") % (2**31)
                await conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)

                # Check for existing booking
                existing = await conn.fetchrow(
                    """SELECT id FROM appointments
                       WHERE doctor_id = $1 AND slot_time = $2 AND status != 'cancelled'""",
                    doctor_id,
                    slot_dt,
                )

                if existing:
                    logger.info("Slot conflict: doctor=%s slot=%s", doctor_id, slot_dt)
                    alternatives = await self.get_available_slots(
                        date_str=slot_dt.date().isoformat(),
                        doctor_name=None,
                        preferred_time="any",
                        limit=3,
                    )
                    return BookingResult(
                        success=False,
                        conflict_reason="That slot is no longer available.",
                        alternatives=alternatives,
                    )

                # Book it
                import uuid
                appt_id = str(uuid.uuid4())
                confirm_code = appt_id[:8].upper()

                await conn.execute(
                    """INSERT INTO appointments
                       (id, patient_id, doctor_id, slot_time, status, reason, confirmation_code, created_at)
                       VALUES ($1, $2, $3, $4, 'confirmed', $5, $6, NOW())""",
                    appt_id,
                    patient_id,
                    doctor_id,
                    slot_dt,
                    reason,
                    confirm_code,
                )

                logger.info(
                    "Appointment booked: id=%s patient=%s doctor=%s slot=%s",
                    appt_id, patient_id, doctor_id, slot_dt,
                )
                return BookingResult(
                    success=True,
                    appointment_id=appt_id,
                    confirmation_code=confirm_code,
                )

    async def reschedule(
        self,
        appointment_id: str,
        new_slot_time: str,
        reason: str = "",
    ) -> BookingResult:
        new_dt = datetime.fromisoformat(new_slot_time)

        async with self.pool.acquire() as conn:
            existing_appt = await conn.fetchrow(
                "SELECT * FROM appointments WHERE id = $1 AND status = 'confirmed'",
                appointment_id,
            )
            if not existing_appt:
                return BookingResult(success=False, conflict_reason="Appointment not found or already cancelled.")

            # Try to book the new slot
            result = await self.book_slot(
                patient_id=existing_appt["patient_id"],
                doctor_id=existing_appt["doctor_id"],
                slot_time=new_slot_time,
                reason=reason or existing_appt["reason"],
            )

            if result.success:
                # Cancel the old appointment
                await conn.execute(
                    "UPDATE appointments SET status = 'rescheduled', updated_at = NOW() WHERE id = $1",
                    appointment_id,
                )
                return result
            else:
                return result

    async def cancel(self, appointment_id: str, reason: str = "") -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE appointments
                   SET status = 'cancelled', cancellation_reason = $2, updated_at = NOW()
                   WHERE id = $1 AND status IN ('confirmed', 'pending')""",
                appointment_id,
                reason,
            )
            cancelled = result.split()[-1] != "0"
            if cancelled:
                logger.info("Appointment cancelled: id=%s reason=%s", appointment_id, reason)
            return cancelled

    async def find_doctors(
        self,
        query: str,
        language_spoken: Optional[str] = None,
    ) -> list[dict]:
        async with self.pool.acquire() as conn:
            lang_filter = "AND $2 = ANY(languages_spoken)" if language_spoken else ""
            args = [f"%{query}%"]
            if language_spoken:
                args.append(language_spoken)

            rows = await conn.fetch(
                f"""SELECT id, name, specialty, languages_spoken
                    FROM doctors
                    WHERE (LOWER(name) LIKE LOWER($1) OR LOWER(specialty) LIKE LOWER($1))
                    {lang_filter}
                    LIMIT 5""",
                *args,
            )
        return [dict(r) for r in rows]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _parse_date(self, date_str: str) -> date:
        now = datetime.now(tz=IST).date()
        match date_str.lower().strip():
            case "today":
                return now
            case "tomorrow":
                return now + timedelta(days=1)
            case "this week":
                return now  # query will span the week
            case _:
                try:
                    return date.fromisoformat(date_str)
                except ValueError:
                    return now + timedelta(days=1)

    def _time_filter(self, preferred_time: str) -> str:
        match preferred_time.lower():
            case "morning":
                return "AND EXTRACT(HOUR FROM s.slot_time AT TIME ZONE 'Asia/Kolkata') BETWEEN 8 AND 12"
            case "afternoon":
                return "AND EXTRACT(HOUR FROM s.slot_time AT TIME ZONE 'Asia/Kolkata') BETWEEN 12 AND 17"
            case "evening":
                return "AND EXTRACT(HOUR FROM s.slot_time AT TIME ZONE 'Asia/Kolkata') BETWEEN 17 AND 20"
            case _:
                return ""

    def _format_slot(self, dt: datetime) -> str:
        """Human-readable slot time for TTS: 'Tuesday, 3 August at 10:30 AM'"""
        local = dt.astimezone(IST)
        return local.strftime("%A, %-d %B at %-I:%M %p")
