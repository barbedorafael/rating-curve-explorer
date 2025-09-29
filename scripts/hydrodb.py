import sqlite3
import pandas as pd

from scipy.optimize import least_squares

class HydroDB:
   def __init__(self, db_path, station_id):
        """
        Initialize DB.

        Parameters
        ----------
        db_path : str
            Path to SQLite database file
        station_id : int
            Station ID for which to adjust rating curve
        """
        self.db_path = db_path
        self.station_id = station_id
        self.stage_discharge_data = None

    def load_stage_discharge_data(self, start_date=None, end_date=None):
        """
        Load stage-discharge measurements from database.

        Parameters
        ----------
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT date, time, level, discharge
        FROM stage_discharge
        WHERE station_id = ?
        AND level IS NOT NULL
        AND discharge IS NOT NULL
        """
        params = [self.station_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date, time"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            raise ValueError(f"No stage-discharge data found for station {self.station_id}")

        # Convert date strings to datetime objects
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

        self.stage_discharge_data = {
            'dates': df['datetime'].values,
            'levels': df['level'].values,  # cm
            'discharges': df['discharge'].values  # m3/s
        }

        return self.stage_discharge_data