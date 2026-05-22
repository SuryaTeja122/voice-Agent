"""
workers/campaign.py

Celery tasks for outbound campaign calls.
Queues: 'campaigns' (high volume), 'reminders' (time-sensitive)

Tasks:
  initiate_campaign_call  — places outbound call via Twilio, then agent joins
  schedule_reminders      — periodic beat task to scan upcoming appointments
  retry_unanswered        — retries calls that weren't answered
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def enqueue_outbound_call(
    patient_id: str,
    campaign_id: str,
    appointment_id: Optional[str],
    call_type: str = "reminder",
) -> object:
    """Public API to enqueue an outbound call task."""
    queue = "reminders" if call_type == "reminder" else "campaigns"
    return initiate_campaign_call.apply_async(
        kwargs={
            "patient_id": patient_id,
            "campaign_id": campaign_id,
            "appointment_id": appointment_id,
            "call_type": call_type,
        },
        queue=queue,
    )


@celery_app.task(
    bind=True,
    name="campaign.initiate_call",
    max_retries=3,
    default_retry_delay=3600,   # retry after 1 hour if unanswered
    queue="campaigns",
)
def initiate_campaign_call(
    self,
    patient_id: str,
    campaign_id: str,
    appointment_id: Optional[str] = None,
    call_type: str = "reminder",
):
    """
    Places an outbound call to a patient.
    1. Fetches patient phone number
    2. Places call via Twilio
    3. Agent WebSocket connects and handles the conversation
    """
    import httpx
    from twilio.rest import Client as TwilioClient

    logger.info(
        "Initiating %s call: patient=%s campaign=%s",
        call_type, patient_id, campaign_id,
    )

    twilio = TwilioClient(
        os.environ["TWILIO_ACCOUNT_SID"],
        os.environ["TWILIO_AUTH_TOKEN"],
    )

    # Fetch patient phone via internal API
    base_url = os.environ.get("AGENT_BASE_URL", "http://localhost:8000")
    try:
        resp = httpx.get(
            f"{base_url}/api/patients/{patient_id}/phone",
            timeout=5,
        )
        resp.raise_for_status()
        phone_number = resp.json()["phone"]
    except Exception as e:
        logger.error("Failed to fetch phone for patient %s: %s", patient_id, e)
        raise self.retry(exc=e)

    # Generate a unique call_id that carries campaign context
    import uuid
    call_id = f"out_{campaign_id}_{patient_id}_{uuid.uuid4().hex[:8]}"

    # TwiML URL — tells Twilio to connect the call to our WebSocket agent
    twiml_url = f"{base_url}/api/twiml/outbound?call_id={call_id}&campaign_id={campaign_id}&call_type={call_type}"

    try:
        call = twilio.calls.create(
            to=phone_number,
            from_=os.environ["TWILIO_PHONE_NUMBER"],
            url=twiml_url,
            status_callback=f"{base_url}/api/calls/status",
            status_callback_method="POST",
            timeout=30,
        )
        logger.info("Twilio call SID=%s placed for patient=%s", call.sid, patient_id)
        return {"call_sid": call.sid, "call_id": call_id}

    except Exception as e:
        logger.error("Twilio call failed for patient %s: %s", patient_id, e)
        raise self.retry(exc=e)


@celery_app.task(
    name="campaign.schedule_reminders",
    queue="reminders",
)
def schedule_reminder_campaign():
    """
    Periodic task (run every 30 minutes via Celery Beat).
    Scans appointments in the next 24 hours and enqueues reminder calls
    for patients who haven't been called yet.
    """
    import httpx

    base_url = os.environ.get("AGENT_BASE_URL", "http://localhost:8000")
    campaign_id = f"reminder_{datetime.utcnow().strftime('%Y%m%d')}"

    try:
        # Fetch upcoming appointments needing reminders
        resp = httpx.get(
            f"{base_url}/api/appointments/upcoming-reminders",
            timeout=10,
        )
        resp.raise_for_status()
        appointments = resp.json()["appointments"]

        logger.info("Scheduling reminders for %d appointments", len(appointments))

        for appt in appointments:
            enqueue_outbound_call(
                patient_id=appt["patient_id"],
                campaign_id=campaign_id,
                appointment_id=appt["id"],
                call_type="reminder",
            )

        return {"scheduled": len(appointments)}

    except Exception as e:
        logger.error("Reminder campaign scheduling failed: %s", e)
        raise


@celery_app.task(
    name="campaign.retry_unanswered",
    queue="campaigns",
)
def retry_unanswered_calls():
    """
    Retries calls that were placed but not answered (Twilio status='no-answer').
    Runs every 2 hours via Celery Beat. Maximum 3 retries per patient per day.
    """
    import httpx

    base_url = os.environ.get("AGENT_BASE_URL", "http://localhost:8000")

    try:
        resp = httpx.get(
            f"{base_url}/api/calls/unanswered",
            params={"max_age_hours": 2, "max_retries": 3},
            timeout=10,
        )
        resp.raise_for_status()
        unanswered = resp.json()["calls"]

        for call in unanswered:
            enqueue_outbound_call(
                patient_id=call["patient_id"],
                campaign_id=call["campaign_id"],
                appointment_id=call.get("appointment_id"),
                call_type=call["call_type"],
            )

        logger.info("Retrying %d unanswered calls", len(unanswered))
        return {"retried": len(unanswered)}

    except Exception as e:
        logger.error("Retry task failed: %s", e)
        raise
