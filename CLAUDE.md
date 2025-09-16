# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rating Curve Explorer is a Python tool for loading, visualizing, and adjusting stage-discharge (rating) curves for river gauging stations. The project processes hydrological data from CSV files into a SQLite database and provides visualization capabilities for timeseries, cross-sections, and rating curves.

## Commands

### Data Loading (ETL)
```bash
python scripts/db_load.py
```
Loads CSV data from `data/<station_id>/` directories into SQLite database using schema.sql.

### Usage Examples
```python
from scripts.hydrodb import HydroDB
db = HydroDB("data/hydrodata.sqlite")

# Timeseries plots
db.plot_timeseries(station=86720000, variable="level", start_date="2024-01-01", end_date="2025-01-01", rolling=7)
db.plot_timeseries(station=86720000, variable="discharge", rolling=30)

# Cross-section profile
db.plot_vertical_profile(survey_id=3875)

# Rating curves with segmentation
db.plot_rating_curve(station=86720000, level_breaks=[100, 300, 600], fit=True)
```

## Architecture

### Core Components

**HydroDB Class** (`scripts/hydrodb.py`): Main interface for data visualization and analysis
- `plot_timeseries()`: Daily level/discharge series with optional rolling averages
- `plot_vertical_profile()`: Cross-section elevation profiles
- `plot_rating_curve()`: Stage-discharge relationships with segmented power-law fitting

**Database Schema** (`schema.sql`): SQLite with foreign key constraints
- `stations`: Station metadata (coordinates, basin info, drainage area)
- `timeseries_cota`: Daily water level series
- `timeseries_vazao`: Daily discharge series
- `stage_discharge`: Instantaneous stage-discharge measurements
- `vertical_profile_survey`: Cross-section survey metadata
- `vertical_profile`: Cross-section measurement points

### Data Flow
1. Raw CSVs in `data/<station_id>/` (station metadata, timeseries, measurements, profiles)
2. ETL via `db_load.py` transforms to normalized SQLite schema
3. HydroDB provides query/visualization interface with matplotlib

### Rating Curve Methodology
- Model: `Q = a * (H - h0)^b` where H=level, Q=discharge
- Segmentation via `level_breaks` parameter creates piecewise fits
- Non-linear least squares with Huber loss (robust to outliers)
- Constraints: `a > 0`, `0.1 ≤ b ≤ 6`, `h0 < min(H)`

## Dependencies

Core: numpy>=1.25, pandas>=2.0, matplotlib>=3.7, scipy>=1.11

No additional build/test/lint tools configured - uses standard Python environment.