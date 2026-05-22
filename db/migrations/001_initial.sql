"""
db/migrations/001_initial.sql

Run via: psql $DATABASE_URL -f db/migrations/001_initial.sql
Or apply through Alembic (see alembic/versions/).
"""

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Patients
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20) NOT NULL UNIQUE,
    language_preference VARCHAR(5) DEFAULT 'en',
    preferred_time_of_day VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Doctors
CREATE TABLE IF NOT EXISTS doctors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    specialty VARCHAR(100) NOT NULL,
    languages_spoken TEXT[] DEFAULT ARRAY['en'],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Doctor availability slots
CREATE TABLE IF NOT EXISTS doctor_availability (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doctor_id UUID NOT NULL REFERENCES doctors(id),
    slot_time TIMESTAMPTZ NOT NULL,
    is_available BOOLEAN DEFAULT TRUE,
    UNIQUE(doctor_id, slot_time)
);
CREATE INDEX idx_avail_doctor_slot ON doctor_availability(doctor_id, slot_time);
CREATE INDEX idx_avail_slot_time ON doctor_availability(slot_time);

-- Appointments
CREATE TYPE appt_status AS ENUM ('pending','confirmed','rescheduled','cancelled','completed');
CREATE TABLE IF NOT EXISTS appointments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id),
    doctor_id UUID NOT NULL REFERENCES doctors(id),
    slot_time TIMESTAMPTZ NOT NULL,
    status appt_status DEFAULT 'confirmed',
    reason TEXT DEFAULT '',
    confirmation_code VARCHAR(16),
    cancellation_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);
CREATE INDEX idx_appt_doctor_slot ON appointments(doctor_id, slot_time) WHERE status != 'cancelled';
CREATE INDEX idx_appt_patient ON appointments(patient_id);

-- Interaction summaries with pgvector embeddings
CREATE TABLE IF NOT EXISTS interaction_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id),
    call_id VARCHAR(100) NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    embedding vector(1536),
    language VARCHAR(5) DEFAULT 'en',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_summary_embedding
    ON interaction_summaries
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Patient preferred doctors (many-to-many via denormalised name for simplicity)
CREATE TABLE IF NOT EXISTS patient_preferred_doctors (
    patient_id UUID NOT NULL REFERENCES patients(id),
    doctor_name VARCHAR(255) NOT NULL,
    PRIMARY KEY (patient_id, doctor_name)
);

-- Campaign rejections
CREATE TABLE IF NOT EXISTS campaign_rejections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id),
    campaign_id VARCHAR(100) NOT NULL,
    reason VARCHAR(255) DEFAULT 'no_reason',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(patient_id, campaign_id)
);

-- Call log (Twilio SID → patient mapping)
CREATE TYPE call_direction AS ENUM ('inbound', 'outbound');
CREATE TABLE IF NOT EXISTS call_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id VARCHAR(100) NOT NULL UNIQUE,
    patient_id UUID REFERENCES patients(id),
    direction call_direction,
    campaign_id VARCHAR(100),
    status VARCHAR(30) DEFAULT 'initiated',
    duration_seconds INTEGER,
    latency_report JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed data: sample doctors
INSERT INTO doctors (name, specialty, languages_spoken) VALUES
    ('Dr. Priya Menon',    'Cardiology',           ARRAY['en','ta']),
    ('Dr. Anand Sharma',   'General Medicine',     ARRAY['en','hi']),
    ('Dr. Kavitha Rajan',  'Gynaecology',          ARRAY['en','ta','hi']),
    ('Dr. Vikram Nair',    'Orthopaedics',         ARRAY['en','ta']),
    ('Dr. Sunita Patel',   'Paediatrics',          ARRAY['en','hi'])
ON CONFLICT DO NOTHING;
