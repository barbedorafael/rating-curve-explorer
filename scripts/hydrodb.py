import sqlite3
import pandas as pd

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

        df['level'] /= 100 # Convert to meters

        df = df[['datetime', 'level', 'discharge']]

        return df

    def load_rating_curve_data(self):
        """
        Load rating curve parameters from database for the station.

        Returns
        -------
        pd.DataFrame
            DataFrame with rating curve parameters including:
            - start_date, end_date: validity period
            - segment_number: segment identifier (XX/YY format)
            - h_min, h_max: height range in cm
            - h0_param, a_param, n_param: curve parameters for Q = a*(H-h0)^n
            - date_inserted: when the record was added
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT start_date, end_date, segment_number, h_min, h_max,
               h0_param, a_param, n_param, date_inserted
        FROM rating_curve
        WHERE station_id = ?
        ORDER BY start_date, segment_number
        """

        df = pd.read_sql_query(query, conn, params=[self.station_id])
        conn.close()

        if df.empty:
            raise ValueError(f"No rating curve data found for station {self.station_id}")

        return df