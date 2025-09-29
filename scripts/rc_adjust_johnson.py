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
import matplotlib.pyplot as plt
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

    def _calculate_johnson_parameters(self, h1, q1, h2, levels_cm, discharges):
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

        Returns
        -------
        tuple
            (h0, n, a)
        """
        # Convert to meters and handle duplicates
        levels_unique, discharges_unique = self._treat_duplicate_levels(
            levels_cm / 100.0, discharges)

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


        return h0, n, a

    def _extract_segment_measurements(self, level_limits):
        """
        Extract measurements for each segment based on level limits and validity period.

        Parameters
        ----------
        level_limits : list
            Level boundaries for segments in meters

        Returns
        -------
        tuple
            (segment_levels, segment_discharges)
        """

        levels = self.stage_discharge_data['levels'] / 100.0 # Convert to meters
        discharges = self.stage_discharge_data['discharges']

        segment_levels, segment_discharges = [], []

        for i in range(len(level_limits) - 1):
            mask = np.where((levels >= level_limits[i]) & 
                            (levels < level_limits[i + 1]))

            segment_levels.append(levels[mask])
            segment_discharges.append(discharges[mask])

        return segment_levels, segment_discharges

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

        continuity_percent = 100 * np.abs(np.nansum([q_cont1, -q_cont2])) / np.nanmean([q_cont1, q_cont2])

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

        # Check parameter count (must be even: n segments need n h-values + n limits)
        if len(X) < 4:
            return np.nan

        if len(X) % 2 != 0:
            raise ValueError("Number of parameters must be even")

        # Calculate number of segments
        num_segments = len(X) // 2 - 1

        # Construct level limits for segments
        level_limits = [level_range[0]]
        level_limits.extend(X[-num_segments:])  # Add intermediate limits
        level_limits.append(level_range[1])     # Add upper limit
        level_limits.append(9999)               # Add in case extrapolation segment falls within some measurement

        # Extract measurements for each segment
        segment_levels, segment_discharges = self._extract_segment_measurements(level_limits)

        # Prepare interpolation data
        levels_unique, discharges_unique = self._treat_duplicate_levels(
            self.stage_discharge_data['levels'] / 100.0,
            self.stage_discharge_data['discharges'])

        # Calculate parameters for each segment
        a_arr, h0_arr, n_arr = [], [], []

        q_calc_arr = []

        # Calculate Johnson parameters for first num_segments-1 segments
        for n in range(num_segments):
            q1 = interp1d(levels_unique, discharges_unique,
                        kind='linear', fill_value="extrapolate")(X[n])

            h0, n_exp, a = self._calculate_johnson_parameters(
                X[n], q1, X[n + 1],
                self.stage_discharge_data['levels'],
                self.stage_discharge_data['discharges']
            )

            q_calc = np.maximum(a*(segment_levels[n])**n_exp, 0)

            a_arr.append(a)
            h0_arr.append(h0)
            n_arr.append(n_exp)
            q_calc_arr.append(q_calc)


        # Add extrapolation segment for the last segment
        a_arr.append(extrapolation_params[0])
        h0_arr.append(extrapolation_params[1])
        n_arr.append(extrapolation_params[2])

        print(a_arr)

        q_calc_arr.append(np.maximum(extrapolation_params[0] * (segment_levels[num_segments - 1] - extrapolation_params[1])**extrapolation_params[2], 0))

        # Convert to arrays
        a_arr = np.asarray(a_arr)
        h0_arr = np.asarray(h0_arr)
        n_arr = np.asarray(n_arr)

        q_calc_all = np.concatenate(q_calc_arr)
        q_obs_all = np.concatenate(segment_discharges)

        # Calculate deviations
        deviations = np.abs(np.sum(q_calc_all, -q_obs_all)) / q_obs_all

        # Calculate continuity errors at segment boundaries
        continuity_errors = []

        # Check continuity between optimized segments
        for n in range(len(level_limits)-1):
            _, _, cont_error = self._calculate_continuity(
                level_limits[n],
                [a_arr[n - 1], h0_arr[n - 1], n_arr[n - 1]],
                [a_arr[n], h0_arr[n], n_arr[n]]
            )
            continuity_errors.append(cont_error)


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
            Number of segments to fit (excluding extrapolation segment)
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

        # Use full level range for optimization
        optimization_range = level_range

        # Setup optimization bounds and constraints
        eps = 1e-6

        # Bounds: [h1, h2, ..., hn, limit1, limit2, ..., limitn]
        bounds = []

        # Add bounds for h values (segment boundaries) - only optimize up to extrapolation start
        for i in range(num_segments):
            bounds.append((optimization_range[0], optimization_range[1]))

        # Add bounds for segment limits - only optimize up to extrapolation start
        for i in range(num_segments):
            bounds.append((optimization_range[0], optimization_range[1]))

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
            disp=True,
            workers=1
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
        num_segments = len(X) // 2

        # Construct level limits
        level_limits = [level_range[0]]
        level_limits.extend(X[-num_segments:])
        level_limits.append(level_range[1])

        # Extract segment measurements
        segment_levels, segment_discharges = self._extract_segment_measurements(level_limits)

        # Calculate parameters for each segment
        levels_unique, discharges_unique = self._treat_duplicate_levels(
            self.stage_discharge_data['levels'] / 100.0,
            self.stage_discharge_data['discharges']
        )

        segments = []

        for n in range(num_segments - 1):
            q1 = interp1d(levels_unique, discharges_unique,
                        kind='linear', fill_value="extrapolate")(X[n])

            h0, n_exp, a = self._calculate_johnson_parameters(
                X[n], q1, X[n + 1],
                self.stage_discharge_data['levels'],
                self.stage_discharge_data['discharges']
            )

            segments.append({
                'segment_number': n + 1,
                'h_min': level_limits[n] * 100,      # Convert to cm
                'h_max': level_limits[n + 1] * 100,  # Convert to cm
                'h0_param': round(h0, 2),
                'a_param': round(a, 4),
                'n_param': round(n_exp, 3),
                'level_range': (level_limits[n], level_limits[n + 1])
            })

        # Add extrapolation segment for the last segment
        segments.append({
            'segment_number': num_segments,
            'h_min': level_limits[num_segments - 1] * 100,  # Start from previous segment end
            'h_max': level_limits[-1] * 100,
            'h0_param': round(extrapolation_params[1], 2),
            'a_param': round(extrapolation_params[0], 4),
            'n_param': round(extrapolation_params[2], 3),
            'level_range': (level_limits[num_segments - 1], level_limits[-1])
        })

        return {
            'station_id': self.station_id,
            'num_segments': num_segments,
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

    def plot_rating_curve(self, figsize=(12, 8), show_segments=True, show_residuals=True):
        """
        Plot the stage-discharge relationship with fitted segments.

        Parameters
        ----------
        figsize : tuple, default (12, 8)
            Figure size (width, height) in inches
        show_segments : bool, default True
            Whether to show individual segments with different colors
        show_residuals : bool, default True
            Whether to show residual plot as subplot

        Returns
        -------
        tuple
            (fig, axes) matplotlib figure and axes objects
        """
        if self.stage_discharge_data is None:
            raise ValueError("No stage-discharge data loaded. Run load_stage_discharge_data() first.")

        if self.optimized_params is None:
            raise ValueError("No optimized parameters available. Run optimization first.")

        # Create subplots
        if show_residuals:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = None

        # Plot observed data
        levels_cm = self.stage_discharge_data['levels']
        discharges_obs = self.stage_discharge_data['discharges']

        ax1.scatter(levels_cm, discharges_obs, alpha=0.6, color='black', s=30,
                   label='Observed', zorder=3)

        # Generate smooth curve for plotting fitted segments
        # Extend range to include extrapolation region if needed
        max_plot_level = max(levels_cm.max(),
                           max([seg['h_max'] for seg in self.optimized_params['segments']]))
        level_range_cm = np.linspace(levels_cm.min(), max_plot_level, 500)
        discharges_fit = self.calculate_rating_curve(level_range_cm)

        if show_segments:
            # Plot each segment with different colors
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.optimized_params['segments'])))

            for i, segment in enumerate(self.optimized_params['segments']):
                # Create level range for this segment
                h_min_m = segment['level_range'][0]
                h_max_m = segment['level_range'][1]

                # Generate points for smooth curve within segment
                h_segment = np.linspace(h_min_m, h_max_m, 100)
                q_segment = self._power_law_equation(
                    segment['a_param'], h_segment,
                    segment['h0_param'], segment['n_param']
                )

                # Convert to cm for plotting
                h_segment_cm = h_segment * 100

                ax1.plot(h_segment_cm, q_segment, color=colors[i], linewidth=2,
                        label=f"Segment {segment['segment_number']}: "
                              f"Q = {segment['a_param']:.2f}×(H-{segment['h0_param']:.2f})^{segment['n_param']:.2f}")

                # Add vertical lines at segment boundaries
                if i < len(self.optimized_params['segments']) - 1:
                    boundary_level_cm = segment['h_max']
                    ax1.axvline(boundary_level_cm, color=colors[i], linestyle='--', alpha=0.7,
                              label=f'Boundary at {boundary_level_cm:.0f} cm' if i == 0 else '')
        else:
            # Plot single fitted curve
            ax1.plot(level_range_cm, discharges_fit, 'r-', linewidth=2,
                    label='Fitted curve', zorder=2)

        # Add extrapolation start indicator
        segments = self.optimized_params['segments']
        if len(segments) > 1:
            # Find where extrapolation starts (last segment should be extrapolation)
            last_optimized_segment = segments[-2] if len(segments) > 1 else segments[-1]
            extrapolation_segment = segments[-1]

            # Check if there's a gap between optimized and extrapolation segments
            if last_optimized_segment['h_max'] != extrapolation_segment['h_min']:
                extrap_start_cm = extrapolation_segment['h_min']
                ax1.axvline(extrap_start_cm, color='red', linestyle=':', linewidth=2, alpha=0.8,
                          label=f'Extrapolation starts at {extrap_start_cm:.0f} cm')

        # Set main plot properties
        ax1.set_xlabel('Water Level (cm)')
        ax1.set_ylabel('Discharge (m³/s)')
        ax1.set_title(f'Rating Curve - Station {self.station_id}\n'
                     f'Johnson Method with {self.optimized_params["num_segments"]} segments')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)

        # Log scale for better visualization
        ax1.set_yscale('log')

        if show_residuals and ax2 is not None:
            # Calculate residuals
            discharges_calc = self.calculate_rating_curve(levels_cm)
            residuals = 100 * (discharges_calc - discharges_obs) / discharges_obs

            # Plot residuals
            ax2.scatter(levels_cm, residuals, alpha=0.6, color='blue', s=20)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='±10%')
            ax2.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='±20%')
            ax2.axhline(y=-20, color='orange', linestyle='--', alpha=0.5)

            ax2.set_xlabel('Water Level (cm)')
            ax2.set_ylabel('Residual (%)')
            ax2.set_title('Residuals (Calculated - Observed)/Observed × 100%')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')

            # Set reasonable y-limits for residuals
            residual_max = np.max(np.abs(residuals))
            ax2.set_ylim(-min(50, residual_max * 1.2), min(50, residual_max * 1.2))

        plt.tight_layout()

        if show_residuals:
            return fig, (ax1, ax2)
        else:
            return fig, ax1