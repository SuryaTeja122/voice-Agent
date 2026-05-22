"""
db/models.py

SQLAlchemy ORM models for asyncpg.
Tables: patients, doctors, doctor_availability, appointments,
        interaction_summaries, campaign_rejections, call_log
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, Float,
    ForeignKey, Enum, ARRAY, Integer, func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
import uuid


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=False, unique=True)
    language_preference = Column(String(5), default="en")
    preferred_time_of_day = Column(String(20))   # morning | afternoon | evening
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    appointments = relationship("Appointment", back_populates="patient")


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    specialty = Column(String(100), nullable=False)
    languages_spoken = Column(ARRAY(String), default=["en"])
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    availability = relationship("DoctorAvailability", back_populates="doctor")
    appointments = relationship("Appointment", back_populates="doctor")


class DoctorAvailability(Base):
    __tablename__ = "doctor_availability"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False)
    slot_time = Column(DateTime(timezone=True), nullable=False)
    is_available = Column(Boolean, default=True)

    doctor = relationship("Doctor", back_populates="availability")


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False)
    slot_time = Column(DateTime(timezone=True), nullable=False)
    status = Column(
        Enum("pending", "confirmed", "rescheduled", "cancelled", "completed", name="appt_status"),
        default="confirmed",
    )
    reason = Column(Text, default="")
    confirmation_code = Column(String(16))
    cancellation_reason = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")


class InteractionSummary(Base):
    """Post-call summaries with pgvector embeddings for semantic retrieval."""
    __tablename__ = "interaction_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    call_id = Column(String(100), nullable=False, unique=True)
    summary = Column(Text, nullable=False)
    # embedding stored as native vector type via raw SQL migration
    language = Column(String(5), default="en")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CampaignRejection(Base):
    __tablename__ = "campaign_rejections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    campaign_id = Column(String(100), nullable=False)
    reason = Column(String(255), default="no_reason")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CallLog(Base):
    """Maps Twilio call SIDs / internal call IDs to patients."""
    __tablename__ = "call_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(String(100), nullable=False, unique=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    direction = Column(Enum("inbound", "outbound", name="call_direction"))
    campaign_id = Column(String(100))
    status = Column(String(30), default="initiated")
    duration_seconds = Column(Integer)
    latency_report = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
