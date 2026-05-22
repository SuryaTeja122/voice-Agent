"""
workers/celery_app.py

Celery + Redis broker configuration.
"""

import os
from celery import Celery
from celery.schedules import crontab

celery_app = Celery(
    "voice_agent",
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    include=["workers.campaign"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,           # re-queue on worker crash
    worker_prefetch_multiplier=1,  # fair dispatch for long-running tasks
    task_routes={
        "campaign.initiate_call": {"queue": "campaigns"},
        "campaign.schedule_reminders": {"queue": "reminders"},
        "campaign.retry_unanswered": {"queue": "campaigns"},
    },
    # Periodic tasks (Celery Beat)
    beat_schedule={
        "schedule-reminders-every-30min": {
            "task": "campaign.schedule_reminders",
            "schedule": crontab(minute="*/30"),
        },
        "retry-unanswered-every-2h": {
            "task": "campaign.retry_unanswered",
            "schedule": crontab(minute=0, hour="*/2"),
        },
    },
)
