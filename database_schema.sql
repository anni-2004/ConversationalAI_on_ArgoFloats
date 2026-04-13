CREATE EXTENSION IF NOT EXISTS postgis;

-- Table for all measurements
CREATE TABLE IF NOT EXISTS argo_profiles (
    id SERIAL PRIMARY KEY,
    float_id VARCHAR(50),
    profile_number INT,
    time TIMESTAMP,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    depth DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    salinity DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_float_id ON argo_profiles(float_id);

-- Metadata table for floats
CREATE TABLE IF NOT EXISTS float_metadata (
    float_id VARCHAR(50) PRIMARY KEY,
    wmo_id VARCHAR(50),
    project_name TEXT,
    institution TEXT,
    date_launched TEXT,
    parameters JSONB
);