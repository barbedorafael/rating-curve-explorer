PRAGMA foreign_keys = ON;

-- =========================
-- Core Stations Table
-- =========================
CREATE TABLE IF NOT EXISTS stations (
    station_id INTEGER PRIMARY KEY,         
    name TEXT,
    station_type TEXT,
    lon REAL,
    lat REAL,
    basin_id INTEGER,
    sub_basin_id INTEGER,
    river_id INTEGER,
    state_id INTEGER,
    municipality_id INTEGER,
    responsible_id INTEGER,
    responsible_unit TEXT,
    responsible_jurisdiction TEXT,
    operator_id INTEGER,
    operator_unit TEXT,
    operator_subunit TEXT,
    additional_code TEXT,
    altitude REAL,
    drainage_area REAL
);

-- =========================
-- Timeseries: Cota
-- =========================
CREATE TABLE IF NOT EXISTS timeseries_cota (
    station_id INTEGER NOT NULL REFERENCES stations(station_id),
    date TEXT NOT NULL,      -- YYYY-MM-DD
    method INTEGER,
    value REAL NOT NULL,
    status INTEGER,
    PRIMARY KEY (station_id, date)
);

-- =========================
-- Timeseries: Vazao
-- =========================
CREATE TABLE IF NOT EXISTS timeseries_vazao (
    station_id INTEGER NOT NULL REFERENCES stations(station_id),
    date TEXT NOT NULL,      -- YYYY-MM-DD
    method INTEGER,
    value REAL NOT NULL,
    status INTEGER,
    PRIMARY KEY (station_id, date)
);

-- =========================
-- Stage-Discharge Measurements
-- =========================
CREATE TABLE IF NOT EXISTS stage_discharge (
    station_id INTEGER NOT NULL REFERENCES stations(station_id),
    date TEXT NOT NULL,      -- YYYY-MM-DD
    time TEXT NOT NULL,      -- HH:MM
    consistency INTEGER,
    level REAL,              -- cm
    discharge REAL,          -- m3/s
    area REAL,               -- m²
    velocity REAL,           -- m/s
    width REAL,              -- cm
    depth REAL,              -- cm (mean depth)
    instrument INTEGER,
    PRIMARY KEY (station_id, date, time)
);

-- =========================
-- Vertical Profile Surveys
-- =========================
CREATE TABLE IF NOT EXISTS vertical_profile_survey (
    survey_id INTEGER PRIMARY KEY,            -- internal unique key
    station_id INTEGER NOT NULL REFERENCES stations(station_id),
    date TEXT NOT NULL,                        -- YYYY-MM-DD
    num_survey INTEGER,                        -- NumLevantamento
    section_type TEXT,                         -- TipoSecao
    num_verticals INTEGER,                     -- NumVerticais
    dist_pipf REAL,                            -- DistanciaPIPF
    dist_max REAL,                             -- EixoXDistMaxima
    dist_min REAL,                             -- EixoXDistMinima
    elev_max REAL,                             -- EixoYCotaMaxima
    elev_min REAL,                             -- EixoYCotaMinima
    step_elev REAL,                            -- ElmGeomPassoCota
    notes TEXT                                 -- Observacoes
);
-- =========================
-- Vertical Profile Measurements
-- =========================
CREATE TABLE IF NOT EXISTS vertical_profile (
    survey_id INTEGER NOT NULL REFERENCES vertical_profile_survey(survey_id),
    distance REAL,                   -- m from reference
    elevation REAL,                  -- cm
    PRIMARY KEY (survey_id, distance)
);
