"""
Rating Curve Adjustment using Johnson Method

This module implements the Johnson method for rating curve segmentation and optimization.
The method uses differential evolution to optimize segment boundaries and fits power-law
equations Q = a*(H-h0)^n for each segment with continuity constraints.
"""

import datetime as dt
import numpy as np
import pandas as pd
import sqlite3
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, NonlinearConstraint
import warnings
warnings.filterwarnings('ignore')


class JohnsonRatingCurveAdjuster:
    """
    Rating curve adjustment using Johnson method for segmented power-law fitting.

    The Johnson method optimizes segment boundaries for piecewise power-law rating curves
    of the form Q = a*(H-h0)^n, ensuring continuity between segments and using
    differential evolution for global optimization.
    """

    def __init__(self, db_path, station_id):
        """
        Initialize the Johnson rating curve adjuster.

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
        self.optimized_params = None
        self.optimization_result = None

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

    def _power_law_equation(self, a, h, h0, n):
        """Power law rating curve equation: Q = a*(H-h0)^n"""
        return a * (h - h0) ** n

    def _treat_duplicate_levels(self, levels, discharges):
        """
        Handle duplicate level measurements by averaging corresponding discharges.

        Parameters
        ----------
        levels : array
            Water levels in meters
        discharges : array
            Discharge measurements in m3/s

        Returns
        -------
        tuple
            (unique_levels, averaged_discharges)
        """
        df = pd.DataFrame({"levels": levels, "discharges": discharges})
        df_grouped = df.groupby("levels", as_index=False).mean()
        return df_grouped["levels"].values, df_grouped["discharges"].values

    def _calculate_johnson_parameters(self, h1, q1, h2, levels_cm, discharges, segment_levels):
        """
        Calculate Johnson method parameters for a segment.

        Parameters
        ----------
        h1, h2 : float
            Level bounds for the segment (meters)
        q1 : float
            Discharge at h1
        levels_cm : array
            All level measurements in cm
        discharges : array
            All discharge measurements
        segment_levels : array
            Level measurements for this segment

        Returns
        -------
        tuple
            (q2, Q3, H3, h0, n, a, q_calc)
        """
        # Convert to meters and handle duplicates
        levels_unique, discharges_unique = self._treat_duplicate_levels(
            levels_cm / 100.0, discharges
        )

        # Interpolate q2 at h2
        q2 = interp1d(levels_unique, discharges_unique,
                     kind='linear', fill_value="extrapolate")(h2)

        # Calculate geometric mean discharge
        Q3 = np.sqrt(q1 * q2)

        # Interpolate corresponding level H3
        H3 = interp1d(discharges, levels_cm / 100.0,
                     kind='linear', fill_value="extrapolate")(Q3)

        # Calculate Johnson parameters
        h0 = (H3**2.0 - h1*h2) / (2.0*H3 - h1 - h2)
        n = (np.log(q2) - np.log(q1)) / (np.log(h2 - h0) - np.log(h1 - h0))
        a = q2 / ((h2 - h0)**n)

        # Calculate fitted discharges for segment
        q_calc = self._power_law_equation(round(a, 4), segment_levels,
                                        round(h0, 2), round(n, 3))

        return q2, Q3, H3, h0, n, a, q_calc

    def _extract_segment_measurements(self, dates, validity_period, level_limits, levels_cm):
        """
        Extract measurements for each segment based on level limits and validity period.

        Parameters
        ----------
        dates : array
            Measurement dates
        validity_period : tuple
            (start_date, end_date) for valid measurements
        level_limits : list
            Level boundaries for segments in meters
        levels_cm : array
            Level measurements in cm

        Returns
        -------
        tuple
            (segment_masks, segment_levels, segment_discharges)
        """
        segment_masks, segment_levels, segment_discharges = [], [], []

        for i in range(len(level_limits) - 1):
            mask = np.where(np.logical_and.reduce((
                dates >= validity_period[0],
                dates <= validity_period[1],
                levels_cm >= level_limits[i] * 100.0,
                levels_cm <= level_limits[i + 1] * 100.0
            )))[0]

            segment_masks.append(mask)
            segment_levels.append(levels_cm[mask] / 100.0)  # Convert to meters
            segment_discharges.append(self.stage_discharge_data['discharges'][mask])

        return segment_masks, segment_levels, segment_discharges

    def _calculate_continuity(self, h_cont, params_1, params_2):
        """
        Calculate continuity error between two segments at boundary level.

        Parameters
        ----------
        h_cont : float
            Boundary level where continuity is checked
        params_1 : list
            [a, h0, n] for first segment
        params_2 : list
            [a, h0, n] for second segment

        Returns
        -------
        tuple
            (q_cont1, q_cont2, continuity_percent)
        """
        q_cont1 = self._power_law_equation(
            round(params_1[0], 4), h_cont, round(params_1[1], 2), round(params_1[2], 3)
        )
        q_cont2 = self._power_law_equation(
            round(params_2[0], 4), h_cont, round(params_2[1], 2), round(params_2[2], 3)
        )

        continuity_percent = 100 * np.abs(q_cont1 - q_cont2) / np.nanmean([q_cont1, q_cont2])

        return q_cont1, q_cont2, continuity_percent

    def _objective_function(self, X, extrapolation_params, validity_period, level_range):
        """
        Objective function for differential evolution optimization.

        Parameters
        ----------
        X : array
            Optimization variables: [h1, h2, ..., hn, limit1, limit2, ...]
        extrapolation_params : list
            [a, h0, n] for extrapolation segment
        validity_period : tuple
            (start_date, end_date) for measurements
        level_range : tuple
            (h_min, h_max) in meters

        Returns
        -------
        float
            Objective function value (lower is better)
        """
        # Optimization parameter limits
        N_LIMIT_LOW = 1.20
        N_LIMIT_HIGH = 2.0
        CONTINUITY_LIMIT = 0.5
        MAX_DEVIATION_LIMIT = 20.0

        # Check parameter count (must be odd: n segments need 2n-1 h-values + n limits)
        if len(X) < 3:
            return np.nan

        if len(X) % 2 != 1:
            raise ValueError("Number of parameters must be odd")

        # Calculate number of segments
        num_segments = (len(X) - 1) // 2

        # Construct level limits for segments
        level_limits = [level_range[0]]
        level_limits.extend(X[-num_segments:])  # Add intermediate limits
        level_limits.append(level_range[1])     # Add upper limit

        # Extract measurements for each segment
        try:
            _, segment_levels, segment_discharges = self._extract_segment_measurements(
                self.stage_discharge_data['dates'], validity_period, level_limits,
                self.stage_discharge_data['levels']
            )
        except:
            return 1E6

        # Prepare interpolation data
        levels_unique, discharges_unique = self._treat_duplicate_levels(
            self.stage_discharge_data['levels'] / 100.0,
            self.stage_discharge_data['discharges']
        )

        # Calculate parameters for each segment
        a_arr, h0_arr, n_arr, q_calc_arr = [], [], [], []

        for n in range(1, len(X) - num_segments):
            try:
                q1 = interp1d(levels_unique, discharges_unique,
                            kind='linear', fill_value="extrapolate")(X[n - 1])

                _, _, _, h0, n_exp, a, q_calc = self._calculate_johnson_parameters(
                    X[n - 1], q1, X[n],
                    self.stage_discharge_data['levels'],
                    self.stage_discharge_data['discharges'],
                    segment_levels[n - 1]
                )

                a_arr.append(a)
                h0_arr.append(h0)
                n_arr.append(n_exp)
                q_calc_arr.append(q_calc)

            except:
                return 1E6

        # Add extrapolation segment
        a_arr.append(extrapolation_params[0])
        h0_arr.append(extrapolation_params[1])
        n_arr.append(extrapolation_params[2])

        try:
            q_calc_extrap = self._power_law_equation(
                round(extrapolation_params[0], 4), segment_levels[-1],
                round(extrapolation_params[1], 2), round(extrapolation_params[2], 3)
            )
            q_calc_arr.append(q_calc_extrap)
        except:
            return 1E6

        # Convert to arrays
        a_arr = np.asarray(a_arr)
        h0_arr = np.asarray(h0_arr)
        n_arr = np.asarray(n_arr)

        try:
            q_calc_all = np.concatenate(q_calc_arr)
            q_obs_all = np.concatenate(segment_discharges)
        except:
            return 1E6

        # Calculate deviations
        deviations = np.abs(q_calc_all - q_obs_all) / q_obs_all

        # Calculate continuity errors
        continuity_errors = []
        for n in range(1, num_segments + 1):
            try:
                _, _, cont_error = self._calculate_continuity(
                    level_limits[n],
                    [a_arr[n - 1], h0_arr[n - 1], n_arr[n - 1]],
                    [a_arr[n], h0_arr[n], n_arr[n]]
                )
                continuity_errors.append(cont_error)
            except:
                return 1E6

        continuity_errors = np.asarray(continuity_errors)

        # Calculate objective function
        objective = np.mean(100.0 * deviations) + np.sum(continuity_errors)

        # Apply penalties
        if np.any(deviations * 100 >= MAX_DEVIATION_LIMIT):
            objective *= np.nanmax(deviations * 100)

        if np.isnan(objective) or np.any(np.isnan(continuity_errors)):
            objective = 1E6

        if np.any(continuity_errors > CONTINUITY_LIMIT):
            objective *= 10000

        if np.any(n_arr < N_LIMIT_LOW) or np.any(n_arr > N_LIMIT_HIGH):
            objective *= 1000

        return objective

    def optimize_segments(self, num_segments, extrapolation_params,
                         validity_period=None, level_range=None,
                         max_iterations=500, population_size=250):
        """
        Optimize rating curve segments using differential evolution.

        Parameters
        ----------
        num_segments : int
            Number of segments to fit
        extrapolation_params : list
            [a, h0, n] for extrapolation segment
        validity_period : tuple, optional
            (start_date, end_date) for measurements. If None, uses all data.
        level_range : tuple, optional
            (h_min, h_max) in meters. If None, inferred from data.
        max_iterations : int, default 500
            Maximum optimization iterations
        population_size : int, default 250
            Population size for differential evolution

        Returns
        -------
        dict
            Optimization results including parameters and statistics
        """
        if self.stage_discharge_data is None:
            raise ValueError("Must load stage-discharge data first")

        # Set default validity period and level range
        if validity_period is None:
            dates = pd.to_datetime(self.stage_discharge_data['dates'])
            validity_period = (dates.min(), dates.max())
        else:
            validity_period = (pd.to_datetime(validity_period[0]),
                             pd.to_datetime(validity_period[1]))

        if level_range is None:
            levels_m = self.stage_discharge_data['levels'] / 100.0
            level_range = (np.min(levels_m), np.max(levels_m))

        # Setup optimization bounds and constraints
        eps = 1e-6

        # Bounds: [h1, h2, ..., hn, limit1, limit2, ..., limitn]
        bounds = []

        # Add bounds for h values (segment boundaries)
        for i in range(num_segments):
            bounds.append((level_range[0], level_range[1]))

        # Add bounds for segment limits
        for i in range(num_segments):
            bounds.append((level_range[0], level_range[1]))

        # Setup constraints
        constraints = []

        # h-value ordering constraints: h1 < h2 < h3 < ...
        for i in range(num_segments - 1):
            constraints.append(
                NonlinearConstraint(lambda x, i=i: x[i+1] - x[i], eps, np.inf)
            )

        # Segment limit constraints: h_i <= limit_i
        for i in range(num_segments):
            constraints.append(
                NonlinearConstraint(
                    lambda x, i=i: x[i] - x[num_segments + i], -np.inf, 0
                )
            )

        # Limit ordering constraints: limit1 < limit2 < limit3 < ...
        for i in range(num_segments - 1):
            constraints.append(
                NonlinearConstraint(
                    lambda x, i=i: x[num_segments + i + 1] - x[num_segments + i],
                    eps, np.inf
                )
            )

        # Run optimization
        result = differential_evolution(
            self._objective_function,
            bounds=bounds,
            args=(extrapolation_params, validity_period, level_range),
            constraints=constraints,
            strategy='rand1bin',
            maxiter=max_iterations,
            popsize=population_size,
            init='latinhypercube',
            tol=0.01,
            mutation=1.5,
            recombination=0.5,
            polish=False,
            disp=True
        )

        self.optimization_result = result

        # Extract optimized parameters
        if result.success or result.fun < np.inf:
            self.optimized_params = self._extract_optimized_parameters(
                result.x, extrapolation_params, validity_period, level_range
            )

        return {
            'success': result.success,
            'objective_value': result.fun,
            'parameters': self.optimized_params,
            'optimization_result': result
        }

    def _extract_optimized_parameters(self, X, extrapolation_params, validity_period, level_range):
        """Extract and format optimized parameters from optimization result."""
        num_segments = (len(X) - 1) // 2

        # Construct level limits
        level_limits = [level_range[0]]
        level_limits.extend(X[-num_segments:])
        level_limits.append(level_range[1])

        # Extract segment measurements
        _, segment_levels, segment_discharges = self._extract_segment_measurements(
            self.stage_discharge_data['dates'], validity_period, level_limits,
            self.stage_discharge_data['levels']
        )

        # Calculate parameters for each segment
        levels_unique, discharges_unique = self._treat_duplicate_levels(
            self.stage_discharge_data['levels'] / 100.0,
            self.stage_discharge_data['discharges']
        )

        segments = []

        for n in range(1, len(X) - num_segments):
            q1 = interp1d(levels_unique, discharges_unique,
                        kind='linear', fill_value="extrapolate")(X[n - 1])

            _, _, _, h0, n_exp, a, q_calc = self._calculate_johnson_parameters(
                X[n - 1], q1, X[n],
                self.stage_discharge_data['levels'],
                self.stage_discharge_data['discharges'],
                segment_levels[n - 1]
            )

            segments.append({
                'segment_number': n,
                'h_min': level_limits[n - 1] * 100,  # Convert to cm
                'h_max': level_limits[n] * 100,      # Convert to cm
                'h0_param': round(h0, 2),
                'a_param': round(a, 4),
                'n_param': round(n_exp, 3),
                'level_range': (X[n - 1], X[n])
            })

        # Add extrapolation segment
        segments.append({
            'segment_number': num_segments + 1,
            'h_min': level_limits[-2] * 100,
            'h_max': level_limits[-1] * 100,
            'h0_param': round(extrapolation_params[1], 2),
            'a_param': round(extrapolation_params[0], 4),
            'n_param': round(extrapolation_params[2], 3),
            'level_range': (level_limits[-2], level_limits[-1])
        })

        return {
            'station_id': self.station_id,
            'num_segments': num_segments + 1,
            'level_limits': level_limits,
            'segments': segments,
            'validity_period': validity_period
        }

    def save_to_database(self, start_date, end_date=None, segment_prefix="01"):
        """
        Save optimized rating curve parameters to database.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
        segment_prefix : str, default "01"
            Prefix for segment numbering (e.g., "01" for first curve)
        """
        if self.optimized_params is None:
            raise ValueError("No optimized parameters to save. Run optimization first.")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert each segment
        for i, segment in enumerate(self.optimized_params['segments'], 1):
            segment_number = f"{segment_prefix}/{i:02d}"

            cursor.execute("""
                INSERT INTO rating_curve
                (station_id, start_date, end_date, segment_number, h_min, h_max,
                 h0_param, a_param, n_param, date_inserted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.station_id,
                start_date,
                end_date,
                segment_number,
                int(segment['h_min']),
                int(segment['h_max']),
                segment['h0_param'],
                segment['a_param'],
                segment['n_param'],
                dt.datetime.now().strftime('%Y-%m-%d')
            ))

        conn.commit()
        conn.close()

    def calculate_rating_curve(self, levels_cm):
        """
        Calculate discharge values for given levels using optimized parameters.

        Parameters
        ----------
        levels_cm : array
            Water levels in cm

        Returns
        -------
        array
            Calculated discharge values in m3/s
        """
        if self.optimized_params is None:
            raise ValueError("No optimized parameters available. Run optimization first.")

        levels_m = np.asarray(levels_cm) / 100.0
        discharges = np.zeros_like(levels_m)

        for segment in self.optimized_params['segments']:
            # Find levels within this segment range
            mask = (levels_m >= segment['level_range'][0]) & (levels_m <= segment['level_range'][1])

            if np.any(mask):
                discharges[mask] = self._power_law_equation(
                    segment['a_param'],
                    levels_m[mask],
                    segment['h0_param'],
                    segment['n_param']
                )

        return discharges

    def get_optimization_summary(self):
        """
        Get summary of optimization results.

        Returns
        -------
        dict
            Summary including parameters, continuity errors, and fit statistics
        """
        if self.optimized_params is None:
            return None

        # Calculate continuity errors
        continuity_errors = []
        segments = self.optimized_params['segments']

        for i in range(len(segments) - 1):
            boundary_level = segments[i]['level_range'][1]

            q1 = self._power_law_equation(
                segments[i]['a_param'], boundary_level,
                segments[i]['h0_param'], segments[i]['n_param']
            )
            q2 = self._power_law_equation(
                segments[i+1]['a_param'], boundary_level,
                segments[i+1]['h0_param'], segments[i+1]['n_param']
            )

            continuity_error = 100 * np.abs(q1 - q2) / np.mean([q1, q2])
            continuity_errors.append(continuity_error)

        return {
            'station_id': self.station_id,
            'objective_value': self.optimization_result.fun if self.optimization_result else None,
            'num_segments': self.optimized_params['num_segments'],
            'segments': segments,
            'continuity_errors': continuity_errors,
            'max_continuity_error': np.max(continuity_errors) if continuity_errors else 0,
            'optimization_success': self.optimization_result.success if self.optimization_result else False
        }