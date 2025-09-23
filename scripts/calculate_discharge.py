#!/usr/bin/env python3
"""
Calculate discharge values using rating curves and load into timeseries_vazao.

This script:
1. Reads rating curve parameters from the rating_curve table
2. Finds corresponding level data from timeseries_cota within date/level constraints
3. Applies the rating curve equation Q = a * (H - H0)^n
4. Inserts calculated discharge values into timeseries_vazao
"""

import sqlite3
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_db(db_path):
    """Connect to SQLite database with foreign key constraints enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def get_rating_curves(conn):
    """Fetch all rating curve parameters."""
    query = """
    SELECT station_id, segment_number, start_date, end_date,
           h_min, h_max, h0_param, a_param, n_param
    FROM rating_curve
    ORDER BY station_id, start_date, segment_number
    """
    return pd.read_sql_query(query, conn)

def get_level_data(conn, station_id, start_date, end_date, h_min, h_max):
    """Get level data for a station within date and level constraints."""
    query = """
    SELECT station_id, date, value as level, status
    FROM timeseries_cota
    WHERE station_id = ?
      AND date >= ?
      AND (? IS NULL OR date <= ?)
      AND value >= ?
      AND value <= ?
    ORDER BY date
    """

    # Handle NULL end_date (current/ongoing curves)
    end_date_param = end_date if end_date else '9999-12-31'

    return pd.read_sql_query(query, conn, params=[
        station_id, start_date, end_date_param, end_date, h_min, h_max
    ])

def calculate_discharge(level, h0, a, n):
    """Calculate discharge using rating curve equation Q = a * (H - H0)^n."""
    # Ensure H > H0 to avoid negative or complex values
    h_diff = max(level/100 - h0, 0) # convert level to m

    discharge = a * h_diff ** n
    discharge = (discharge*100).astype(int)/100 # truncate to 2 decimal places

    return discharge

def insert_discharge_data(conn, discharge_data):
    """Insert calculated discharge data into timeseries_vazao."""
    if discharge_data.empty:
        return 0

    # Prepare data for insertion
    insert_query = """
    INSERT OR REPLACE INTO timeseries_vazao
    (station_id, date, method, value, status)
    VALUES (?, ?, ?, ?, ?)
    """

    records = []
    for _, row in discharge_data.iterrows():
        if not np.isnan(row['discharge']):
            records.append((
                int(row['station_id']),
                row['date'],
                1,    # method: calculated from rating curve
                float(row['discharge']),
                row['status']    # status: same as level
            ))

    if records:
        conn.executemany(insert_query, records)
        conn.commit()
        logger.info(f"Inserted {len(records)} discharge records")
        return len(records)

    return 0

def process_rating_curves(db_path):
    """Main function to process all rating curves and calculate discharge."""
    conn = connect_db(db_path)

    try:
        # Get all rating curves
        rating_curves = get_rating_curves(conn)
        logger.info(f"Found {len(rating_curves)} rating curve segments")

        if rating_curves.empty:
            logger.warning("No rating curves found in database")
            return

        total_records = 0

        # Process each rating curve
        for _, curve in rating_curves.iterrows():
            logger.info(f"Processing station {curve['station_id']}, segment {curve['segment_number']}")

            # Get level data within constraints
            level_data = get_level_data(
                conn,
                curve['station_id'],
                curve['start_date'],
                curve['end_date'],
                curve['h_min'],
                curve['h_max']
            )
            
            if level_data.empty:
                logger.info(f"No level data found for station {curve['station_id']}, segment {curve['segment_number']}")
                continue

            logger.info(f"Found {len(level_data)} level records for station {curve['station_id']}")

            # Calculate discharge using rating curve
            level_data['discharge'] = calculate_discharge(
                level_data['level'],
                curve['h0_param'],
                curve['a_param'],
                curve['n_param']
            )
            
            # Remove invalid calculations
            valid_data = level_data[~np.isnan(level_data['discharge'])].copy()

            if valid_data.empty:
                logger.warning(f"No valid discharge calculations for station {curve['station_id']}, segment {curve['segment_number']}")
                continue

            logger.info(f"Calculated {len(valid_data)} valid discharge values")
            
            # Insert into database
            records_inserted = insert_discharge_data(conn, valid_data)
            total_records += records_inserted

            logger.info(f"Station {curve['station_id']}, segment {curve['segment_number']}: {records_inserted} records inserted")

        logger.info(f"Processing complete. Total records inserted: {total_records}")

    except Exception as e:
        logger.error(f"Error processing rating curves: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def main():
    """Main entry point."""
    db_path = "data/hydrodata.sqlite"

    logger.info("Starting discharge calculation from rating curves")
    logger.info(f"Database: {db_path}")

    try:
        process_rating_curves(db_path)
        logger.info("Discharge calculation completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main()