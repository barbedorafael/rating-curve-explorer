# Input data layout (extracted from .mdb files)

- `data/<station_id>/_Estacao_.csv` — metadados da estação
- `data/<station_id>/_Cotas_.csv` — série temporal de cotas em formato largo
- `data/<station_id>/_Vazoes_.csv` — série temporal de vazões em formato largo
- `data/<station_id>/_ResumoDescarga_.csv` — medições pontuais (nivel, vazão, etc.)
- `data/<station_id>/_CurvaDescarga_.csv` — validade e parâmetros das curva-chave
- `data/<station_id>/_PerfilTransversal_.csv`, — metadados das medições de seção transversal
- `data/<station_id>/_PerfilTransversalVertical_.csv`, — medições de seção transversal

# Database Schema (SQLite)

SQLite with `PRAGMA foreign_keys = ON;`.
All dates stored as `TEXT` in `YYYY-MM-DD` and times as `TEXT` in `HH:MM`.

This document mirrors the current `schema.sql`. If column units or names in the raw CSVs differ by station or provider, normalize in ETL or document exceptions here.

---

## Tables and Relationships

```
stations (1) ──< timeseries_cota
        │
        ├──< timeseries_vazao
        │
        ├──< stage_discharge
        │
        └──< vertical_profile_survey (1) ──< vertical_profile
```

---

## stations

Metadata for each gauging station.

| Column                    | Type    | Null | Key | Description / Units                                 |
| ------------------------- | ------- | ---- | --- | --------------------------------------------------- |
| station\_id               | INTEGER | NO   | PK  | Unique station code (primary key).                  |
| name                      | TEXT    | YES  |     | Station name.                                       |
| station\_type             | TEXT    | YES  |     | Station type/class.                                 |
| lon                       | REAL    | YES  |     | Longitude (decimal degrees).                        |
| lat                       | REAL    | YES  |     | Latitude (decimal degrees).                         |
| basin\_id                 | INTEGER | YES  |     | Basin code.                                         |
| sub\_basin\_id            | INTEGER | YES  |     | Sub‑basin code.                                     |
| river\_id                 | INTEGER | YES  |     | River code.                                         |
| state\_id                 | INTEGER | YES  |     | State code.                                         |
| municipality\_id          | INTEGER | YES  |     | Municipality code.                                  |
| responsible\_id           | INTEGER | YES  |     | Responsible entity code.                            |
| responsible\_unit         | TEXT    | YES  |     | Responsible unit.                                   |
| responsible\_jurisdiction | TEXT    | YES  |     | Responsible jurisdiction.                           |
| operator\_id              | INTEGER | YES  |     | Operator code.                                      |
| operator\_unit            | TEXT    | YES  |     | Operator unit.                                      |
| operator\_subunit         | TEXT    | YES  |     | Operator sub‑unit.                                  |
| additional\_code          | TEXT    | YES  |     | Additional station code.                            |
| altitude                  | REAL    | YES  |     | Altitude (m).                                       |
| drainage\_area            | REAL    | YES  |     | Drainage area (km² or m² — confirm unit in source). |

---

## timeseries\_cota

Daily water level series (one row per station per date).

| Column      | Type    | Null | Key                        | Description / Units                                |
| ----------- | ------- | ---- | -------------------------- | -------------------------------------------------- |
| station\_id | INTEGER | NO   | FK → stations(station\_id) | Station.                                           |
| date        | TEXT    | NO   | PK (station\_id, date)     | Day (`YYYY-MM-DD`).                                |
| method      | INTEGER | YES  |                            | Measurement/processing method (code).              |
| value       | REAL    | NO   |                            | Water level value (same unit as source; often cm). |
| status      | INTEGER | YES  |                            | Status/flag code.                                  |

Constraint: `PRIMARY KEY (station_id, date)`.

---

## timeseries\_vazao

Daily discharge series (one row per station per date).

| Column      | Type    | Null | Key                        | Description / Units                   |
| ----------- | ------- | ---- | -------------------------- | ------------------------------------- |
| station\_id | INTEGER | NO   | FK → stations(station\_id) | Station.                              |
| date        | TEXT    | NO   | PK (station\_id, date)     | Day (`YYYY-MM-DD`).                   |
| method      | INTEGER | YES  |                            | Measurement/processing method (code). |
| value       | REAL    | NO   |                            | Discharge (m³/s).                     |
| status      | INTEGER | YES  |                            | Status/flag code.                     |

Constraint: `PRIMARY KEY (station_id, date)`.

---

## stage\_discharge

Instantaneous stage–discharge measurements (spot gauging sessions).

| Column      | Type    | Null | Key                        | Description / Units                  |
| ----------- | ------- | ---- | -------------------------- | ------------------------------------ |
| station\_id | INTEGER | NO   | FK → stations(station\_id) | Station.                             |
| date        | TEXT    | NO   | PK part                    | Date (`YYYY-MM-DD`).                 |
| time        | TEXT    | NO   | PK part                    | Time (`HH:MM`).                      |
| consistency | INTEGER | YES  |                            | Consistency/quality code (optional). |
| level       | REAL    | YES  |                            | Water level (cm).                    |
| discharge   | REAL    | YES  |                            | Discharge (m³/s).                    |
| area        | REAL    | YES  |                            | Wet area (m²).                       |
| velocity    | REAL    | YES  |                            | Mean velocity (m/s).                 |
| width       | REAL    | YES  |                            | Section width (cm).                  |
| depth       | REAL    | YES  |                            | Mean depth (cm).                     |
| instrument  | INTEGER | YES  |                            | Instrument code.                     |

Constraint: `PRIMARY KEY (station_id, date, time)`.

---

## vertical\_profile\_survey

Metadata for a cross‑section survey at a station.

| Column         | Type    | Null | Key                        | Description / Units         |
| -------------- | ------- | ---- | -------------------------- | --------------------------- |
| survey\_id     | INTEGER | NO   | PK                         | Internal unique survey id.  |
| station\_id    | INTEGER | NO   | FK → stations(station\_id) | Station.                    |
| date           | TEXT    | NO   |                            | Survey date (`YYYY-MM-DD`). |
| num\_survey    | INTEGER | YES  |                            | NumLevantamento.            |
| section\_type  | TEXT    | YES  |                            | TipoSecao.                  |
| num\_verticals | INTEGER | YES  |                            | NumVerticais.               |
| dist\_pipf     | REAL    | YES  |                            | Distância ao PIPF.          |
| dist\_max      | REAL    | YES  |                            | EixoXDistMaxima.            |
| dist\_min      | REAL    | YES  |                            | EixoXDistMinima.            |
| elev\_max      | REAL    | YES  |                            | EixoYCotaMaxima.            |
| elev\_min      | REAL    | YES  |                            | EixoYCotaMinima.            |
| step\_elev     | REAL    | YES  |                            | ElmGeomPassoCota.           |
| notes          | TEXT    | YES  |                            | Observações.                |

---

## vertical\_profile

Measured points of a cross‑section profile for a given survey.

| Column     | Type    | Null | Key                                        | Description / Units                     |
| ---------- | ------- | ---- | ------------------------------------------ | --------------------------------------- |
| survey\_id | INTEGER | NO   | FK → vertical\_profile\_survey(survey\_id) | Survey.                                 |
| distance   | REAL    | NO   | PK part                                    | Horizontal distance from reference (m). |
| elevation  | REAL    | YES  |                                            | Elevation (m).                          |

Constraint: `PRIMARY KEY (survey_id, distance)`.

---

## Notes & Conventions

* Foreign keys are enforced (`PRAGMA foreign_keys = ON`).
* Dates stored as `TEXT` allow easy filtering with `BETWEEN`, `>=`, `<` when strings follow ISO format.
* Units:

  * `stage_discharge.level`, `width`, `depth` were set in centimeters per source; convert as needed in analytics.
  * `discharge` in m³/s; `area` m²; `velocity` m/s.
* Suggested indexes (optional, for speed with large volumes):

  * `CREATE INDEX idx_cota_station_date ON timeseries_cota(station_id, date);`
  * `CREATE INDEX idx_vazao_station_date ON timeseries_vazao(station_id, date);`
  * `CREATE INDEX idx_sd_station_date ON stage_discharge(station_id, date);`
  * `CREATE INDEX idx_vp_survey ON vertical_profile(survey_id);`


