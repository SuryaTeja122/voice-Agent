"""
services/patient.py

Patient profile management: CRUD, language preference, history, campaign rejection logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class PatientProfile:
    id: str
    name: str
    phone: str
    language_preference: str
    last_visit_summary: Optional[str]
    preferred_doctors: list[str]
    preferred_time_of_day: Optional[str]


class PatientService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_profile(self, patient_id: str) -> Optional[PatientProfile]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT p.id, p.name, p.phone, p.language_preference,
                          p.preferred_time_of_day,
                          ARRAY_AGG(DISTINCT pd.doctor_name) FILTER (WHERE pd.doctor_name IS NOT NULL) AS preferred_doctors,
                          (
                              SELECT summary FROM interaction_summaries
                              WHERE patient_id = p.id
                              ORDER BY created_at DESC LIMIT 1
                          ) AS last_visit_summary
                   FROM patients p
                   LEFT JOIN patient_preferred_doctors pd ON pd.patient_id = p.id
                   WHERE p.id = $1
                   GROUP BY p.id""",
                patient_id,
            )
            if not row:
                return None
            return PatientProfile(
                id=row["id"],
                name=row["name"],
                phone=row["phone"],
                language_preference=row["language_preference"] or "en",
                last_visit_summary=row["last_visit_summary"],
                preferred_doctors=list(row["preferred_doctors"] or []),
                preferred_time_of_day=row["preferred_time_of_day"],
            )

    async def get_appointments(
        self,
        patient_id: str,
        include_past: bool = False,
    ) -> list[dict]:
        async with self.pool.acquire() as conn:
            condition = "" if include_past else "AND a.slot_time > NOW()"
            rows = await conn.fetch(
                f"""SELECT a.id, a.slot_time, a.status, a.reason,
                           a.confirmation_code, d.name AS doctor_name, d.specialty
                    FROM appointments a
                    JOIN doctors d ON a.doctor_id = d.id
                    WHERE a.patient_id = $1 {condition}
                    ORDER BY a.slot_time DESC
                    LIMIT 10""",
                patient_id,
            )
            return [
                {
                    "id": r["id"],
                    "slot_time": r["slot_time"].isoformat(),
                    "slot_display": r["slot_time"].strftime("%A, %-d %B at %-I:%M %p"),
                    "status": r["status"],
                    "reason": r["reason"],
                    "confirmation_code": r["confirmation_code"],
                    "doctor_name": r["doctor_name"],
                    "specialty": r["specialty"],
                }
                for r in rows
            ]

    async def get_language_preference(self, patient_id: str) -> Optional[str]:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT language_preference FROM patients WHERE id = $1",
                patient_id,
            )

    async def update_language_preference(self, patient_id: str, language: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE patients SET language_preference = $2, updated_at = NOW() WHERE id = $1",
                patient_id,
                language,
            )

    async def lookup_by_call_id(self, call_id: str) -> Optional[str]:
        """
        Maps a Twilio call SID or internal call_id to a patient_id.
        For inbound calls, phone number lookup is used.
        """
        async with self.pool.acquire() as conn:
            # Check call_log table (populated by Twilio webhook on call start)
            return await conn.fetchval(
                "SELECT patient_id FROM call_log WHERE call_id = $1 LIMIT 1",
                call_id,
            )

    async def log_campaign_rejection(
        self,
        patient_id: str,
        campaign_id: str,
        reason: str = "no_reason",
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO campaign_rejections (patient_id, campaign_id, reason, created_at)
                   VALUES ($1, $2, $3, NOW())
                   ON CONFLICT (patient_id, campaign_id) DO UPDATE SET reason = $3, created_at = NOW()""",
                patient_id,
                campaign_id,
                reason,
            )
        logger.info("Campaign rejection logged: patient=%s campaign=%s", patient_id, campaign_id)
