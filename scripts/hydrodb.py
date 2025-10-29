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
        self.conn = sqlite3.connect(db_path)
        self.station_id = station_id
        self.stage_discharge_data = None
    
    def load_timeseries_data(self, type='cota', start_date=None, end_date=None):
        """
        Load time-series measurements from database.

        Parameters
        ----------
        type: 'cota' or 'vazao'
            default: 'cota'
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
        """

        query = f"""
        SELECT date, value
        FROM timeseries_{type}
        WHERE station_id = ?
        """
        params = [self.station_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, self.conn, params=params)

        if df.empty:
            raise ValueError(f"No time-series data found for station {self.station_id}")

        if type == 'cota':
            df['value'] /= 100 # Convert to meters

        df = df[['date', 'value']]

        return df
    
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

        query = """
        SELECT date, time, level, discharge
        FROM stage_discharge
        WHERE station_id = ?
        """
        params = [self.station_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date, time"

        df = pd.read_sql_query(query, self.conn, params=params)

        if df.empty:
            raise ValueError(f"No stage-discharge data found for station {self.station_id}")

        df['level'] /= 100 # Convert to meters

        df = df[['date', 'level', 'discharge']]

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
            - a_param, h0_param, n_param: curve parameters for Q = a*(H-h0)^n
            - date_inserted: when the record was added
        """

        query = """
        SELECT start_date, end_date, segment_number, h_min, h_max,
               a_param, h0_param, n_param, date_inserted
        FROM rating_curve
        WHERE station_id = ?
        ORDER BY start_date, segment_number
        """

        df = pd.read_sql_query(query, self.conn, params=[self.station_id])

        if df.empty:
            raise ValueError(f"No rating curve data found for station {self.station_id}")

        return df