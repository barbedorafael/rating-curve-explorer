import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import huber
import matplotlib.pyplot as plt
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as colors
import pandas as pd

class Segment:
    """Represents a power function segment"""
    def __init__(self, a, x0, n, x_start, x_end, cid='new'):
        self.a = round(a, 4)      # 4 decimal places
        self.x0 = round(x0, 2)    # 2 decimal places
        self.n = round(n, 3)      # 3 decimal places
        self.x_start = round(x_start, 2)    # 2 decimal places
        self.x_end = round(x_end, 2)        # 2 decimal places
        self.cid = cid            # curve id for grouping segments
    
    def evaluate(self, x):
        """Evaluate the power function at given x values"""
        return self.a * np.maximum(x - self.x0, 0) ** self.n


class RatingCurveFitter:
    def __init__(self, data, 
                 existing_curves=None, 
                 x_min=None, x_max=None,
                 last_segment_params=None,
                 fixed_breakpoints=None):
        """
        Initialize the rating curve optimization engine.

        This class handles the complete rating curve workflow: data management,
        optimization of segmented power-law curves with advanced penalty functions
        for continuity, crossing prevention, and bias control.

        Parameters
        ----------
        data : pd.DataFrame()
            Table with measurement data points
            Column formats:
                datetime: datetime, YYYY-MM-DD HH:mm
                water level: float, m
                discharge: float, m3/s
        existing_curves : list of Segment objects, optional
            Pre-existing curve segments to include
        x_min, x_max : float, optional
            Domain boundaries. If not provided, computed from data

        Notes
        -----
        Combines data management and optimization in a single class for simplicity.
        """
        
        self.dates = np.array(data.date)
        self.x_data = np.array(data.level)
        self.y_data = np.array(data.discharge)
        self.x_min = x_min if x_min is not None else self.x_data.min()
        self.x_max = x_max if x_max is not None else self.x_data.max()
        
        self.existing_curves = existing_curves or []
        self.last_segment_params = last_segment_params or []
        self.fixed_breakpoints = fixed_breakpoints or []

        # Sort data by x
        idx = np.argsort(self.x_data)
        self.x_data = self.x_data[idx]
        self.y_data = self.y_data[idx]
        self.dates = self.dates[idx]

    def load_rcs(self, df):
        """
        Load rating curve data from DataFrame and convert to Segment objects.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing rating curve parameters with columns:
            - h_min, h_max : float (water levels in cm)
            - a_param, h0_param, n_param : float (power law parameters)
            - start_date, end_date : str (validity period for curve ID)

        Notes
        -----
        Water levels are automatically converted from cm to meters.
        Curve ID is created from start_date and end_date.
        """

        for _, row in df.iterrows():
            # Convert h_min, h_max from cm to meters
            x_start = row['h_min'] / 100.0
            x_end = row['h_max'] / 100.0

            segment = Segment(
                a=row['a_param'],
                x0=row['h0_param'],
                n=row['n_param'],
                x_start=x_start,
                x_end=x_end,
                cid=f"{row['start_date']}_{row['end_date']}"
            )
            self.existing_curves.append(segment)

    def create_objective_function(self,
                                  loss_weight=10,
                                  continuity_threshold=0.01,
                                  continuity_weight=100,
                                  min_segment_range=0.1,
                                  min_range_weight=1,
                                  min_points_per_segment=2,
                                  min_points_weight=1,
                                  param_deviation_weight=1,
                                  n_deviation_weight=2,
                                  bias_weight=5,
                                  n_bias_bins=5,
                                  curve_crossing_weight=10):
        """
        Create a multi-criteria objective function for rating curve optimization.

        Constructs a sophisticated objective function that balances multiple goals:
        data fitting, curve continuity, segment validity, bias control, and
        crossing prevention with existing curves.

        Parameters
        ----------
        loss_weight : float, default 10
            Weight for primary loss function (Mean Percentage Error)
        continuity_threshold : float, default 0.01
            Maximum allowed relative discontinuity at breakpoints (1%)
        continuity_weight : float, default 100
            Penalty weight for continuity violations
        min_segment_range : float, default 0.1
            Minimum x-range each segment must cover (prevents degenerate segments)
        min_range_weight : float, default 1
            Penalty weight for segments below minimum range
        min_points_per_segment : int, default 2
            Minimum data points required per segment
        min_points_weight : float, default 1
            Penalty weight for segments with insufficient data points
        param_deviation_weight : float, default 1
            Penalty for parameter deviations from reference values
        n_deviation_weight : float, default 2
            Additional penalty specifically for exponent (n) parameter deviations
        bias_weight : float, default 5
            Weight for bias penalty to balance errors across domain
        n_bias_bins : int, default 5
            Number of bins for bias calculation using percentile-based binning
        curve_crossing_weight : float, default 20
            Penalty weight for preventing intersection with existing curves

        Returns
        -------
        callable
            Objective function that takes (params, n_segments, return_components=False)
            and returns scalar cost or detailed component breakdown
        """
        
        def objective(params, n_segments, return_components=False):
            segments = self.params_to_segments(params, n_segments)
            
            # Compute fitted values
            y_fitted = np.zeros_like(self.y_data)
            for i, x in enumerate(self.x_data):
                for seg in segments:
                    if seg.x_start <= x <= seg.x_end:
                        y_fitted[i] = seg.evaluate(x)
                        break
            
            # Loss penalty
            residuals = self.y_data - y_fitted
            rel_errors = np.abs(residuals) / (np.abs(self.y_data) + 1e-10)
            loss_penalty = np.nanmean(rel_errors)
            
            # Continuity penalty
            continuity_penalty = 0
            for i in range(len(segments) - 1):
                x_break = segments[i].x_end
                val1 = segments[i].evaluate(x_break)
                val2 = segments[i+1].evaluate(x_break)
                rel_diff = abs(val1 - val2) / (abs(val1) + 1e-10)
                if rel_diff > continuity_threshold:  # 5% threshold
                    continuity_penalty += (rel_diff - continuity_threshold)**2
            
            # Minimum segment range penalty
            min_range_penalty = 0
            for seg in segments:
                segment_range = seg.x_end - seg.x_start
                if segment_range < min_segment_range:
                    # Quadratic penalty for segments that are too narrow
                    min_range_penalty += ((min_segment_range - segment_range) / min_segment_range)**2
            
            # Minimum points per segment penalty
            min_points_penalty = 0
            segment_point_counts = []
            for seg in segments:
                # Count points in this segment
                points_in_segment = np.sum((self.x_data >= seg.x_start) &
                                          (self.x_data <= seg.x_end))
                segment_point_counts.append(points_in_segment)

                if points_in_segment < min_points_per_segment:
                    # Penalty for having too few points
                    min_points_penalty += 1

            # Parameter deviation penalty
            param_deviation_penalty = 0
            n_deviation_penalty = 0
            if self.last_segment_params and len(segments) > 1:
                # Reference parameters from last segment
                ref_a = self.last_segment_params['a']
                ref_x0 = self.last_segment_params['x0']
                ref_n = self.last_segment_params['n']

                # Calculate deviations for all segments except the last one
                for seg in segments[:-1]:
                    # Relative deviations for a and x0
                    a_dev = abs(seg.a - ref_a) / (abs(ref_a) + 1e-10)
                    x0_dev = abs(seg.x0 - ref_x0) / (abs(ref_x0) + 1e-10)

                    param_deviation_penalty += a_dev**2 + x0_dev**2

                    # Special penalty for n parameter deviations
                    n_dev = abs(seg.n - ref_n) / (abs(ref_n) + 1e-10)
                    n_deviation_penalty += n_dev**2

            # Bias penalty - balance errors across x-values using percentile-based bins
            bias_penalty = 0
            if len(self.x_data) >= n_bias_bins:
                # Create bins using data percentiles for equal sample sizes
                percentiles = np.linspace(0, 100, n_bias_bins + 1)
                bin_edges = np.percentile(self.x_data, percentiles)

                # Ensure unique bin edges
                bin_edges = np.unique(bin_edges)
                n_actual_bins = len(bin_edges) - 1

                if n_actual_bins > 1:
                    # Assign data points to bins
                    bin_indices = np.digitize(self.x_data, bin_edges) - 1
                    bin_indices = np.clip(bin_indices, 0, n_actual_bins - 1)

                    # Calculate bias in each bin
                    for i in range(n_actual_bins):
                        mask = (bin_indices == i)
                        if np.sum(mask) > 0:
                            bin_residuals = rel_errors[mask]  # Use relative errors for bias
                            bin_bias = np.mean(bin_residuals - np.mean(rel_errors))  # Deviation from overall mean
                            bias_penalty += bin_bias**2

            # Curve crossing penalty - prevent new curve from crossing existing curves
            curve_crossing_penalty = 0
            if self.existing_curves:
                # Crossing detection: check transition points
                x_check = np.linspace(self.x_min, self.x_max, 100)

                # Track relative positions to detect actual crossings
                for i in range(len(x_check) - 1):
                    x1, x2 = x_check[i], x_check[i + 1]

                    # Skip very small intervals
                    if x2 - x1 < 1e-6:
                        continue

                    # Get values at both endpoints for new curve
                    new_y1 = new_y2 = None
                    for seg in segments:
                        if seg.x_start <= x1 <= seg.x_end:
                            new_y1 = seg.evaluate(x1)
                        if seg.x_start <= x2 <= seg.x_end:
                            new_y2 = seg.evaluate(x2)

                    # Get values at both endpoints for each existing curve
                    for existing_seg in self.existing_curves:
                        existing_y1 = existing_y2 = None
                        if existing_seg.x_start <= x1 <= existing_seg.x_end:
                            existing_y1 = existing_seg.evaluate(x1)
                        if existing_seg.x_start <= x2 <= existing_seg.x_end:
                            existing_y2 = existing_seg.evaluate(x2)

                        # Check for crossing: curves switch relative positions
                        if (new_y1 is not None and new_y2 is not None and
                            existing_y1 is not None and existing_y2 is not None):

                            # Check if the relative ordering changes (indicating a crossing)
                            diff1 = new_y1 - existing_y1  # >0 means new curve is above at x1
                            diff2 = new_y2 - existing_y2  # >0 means new curve is above at x2

                            # If signs are different, curves crossed in this interval
                            if diff1 * diff2 < 0:  # Different signs indicate crossing
                                # Add penalty proportional to the interval size
                                total_diff = abs(diff1) + abs(diff2)
                                curve_crossing_penalty += total_diff / (self.x_max - self.x_min)

            # Total objective
            total = (loss_weight * loss_penalty +
                    continuity_weight * continuity_penalty +
                    min_range_weight * min_range_penalty +
                    min_points_weight * min_points_penalty +
                    param_deviation_weight * param_deviation_penalty +
                    n_deviation_weight * n_deviation_penalty +
                    bias_weight * bias_penalty +
                    curve_crossing_weight * curve_crossing_penalty)

            if return_components:
                return {
                    'total': total,
                    'base_loss': loss_penalty,
                    'continuity_penalty': continuity_penalty,
                    'min_range_penalty': min_range_penalty,
                    'min_points_penalty': min_points_penalty,
                    'param_deviation_penalty': param_deviation_penalty,
                    'n_deviation_penalty': n_deviation_penalty,
                    'bias_penalty': bias_penalty,
                    'curve_crossing_penalty': curve_crossing_penalty,
                    'segment_point_counts': segment_point_counts
                }
            
            return total
        
        return objective
    
    def params_to_segments(self, params, n_segments):
        """
        Convert optimization parameter array to Segment objects.

        Handles both fixed and variable breakpoints, assembling the complete
        segmented curve from the optimized parameters.

        Parameters
        ----------
        params : array-like
            Flattened parameter array from optimizer containing:
            - Variable breakpoints (if any)
            - Segment parameters (a, x0, n) for each optimized segment
        n_segments : int
            Total number of segments in the curve

        Returns
        -------
        list of Segment
            Complete list of segments with proper boundaries and parameters
        """
        segments = []

        if n_segments == 1:
            # Single segment case
            a, x0, n = params
            segments.append(Segment(a, x0, n, self.x_min, self.x_max))
        else:
            # Handle fixed and variable breakpoints
            n_variable_breakpoints = n_segments - 1 - len(self.fixed_breakpoints)

            if n_variable_breakpoints > 0:
                # Extract variable breakpoints
                variable_breakpoints = sorted(params[:n_variable_breakpoints])
            else:
                variable_breakpoints = []

            # Combine fixed and variable breakpoints
            all_breakpoints = sorted(self.fixed_breakpoints + variable_breakpoints)

            # Add boundaries
            all_breaks = [self.x_min] + all_breakpoints + [self.x_max]

            # Extract segment parameters
            param_idx = n_variable_breakpoints
            for i in range(n_segments):
                if self.last_segment_params and i == n_segments - 1:
                    # Use predefined last segment
                    seg = Segment(
                        self.last_segment_params['a'],
                        self.last_segment_params['x0'],
                        self.last_segment_params['n'],
                        all_breaks[i],
                        all_breaks[i+1]
                    )
                else:
                    a = params[param_idx]
                    x0 = params[param_idx + 1]
                    n = params[param_idx + 2]
                    seg = Segment(a, x0, n, all_breaks[i], all_breaks[i+1])
                    param_idx += 3

                segments.append(seg)

        return segments
    
    def get_initial_guess(self, n_segments):
        """
        Generate intelligent initial parameter guesses for optimization.

        Creates reasonable starting points for breakpoints and segment parameters
        while respecting fixed breakpoints and predefined segments.

        Parameters
        ----------
        n_segments : int
            Number of segments to create initial guess for

        Returns
        -------
        numpy.ndarray
            Flattened parameter array ready for optimization
        """
        params = []

        if n_segments > 1:
            # Handle variable breakpoints only (fixed ones are not in the optimization variables)
            n_variable_breakpoints = n_segments - 1 - len(self.fixed_breakpoints)

            if n_variable_breakpoints > 0:
                # Generate initial guesses for variable breakpoints
                # We need to place them avoiding fixed breakpoints
                all_space = np.linspace(self.x_min, self.x_max, n_segments + 1)[1:-1]

                # Remove fixed breakpoints from consideration and adjust
                available_positions = []
                for pos in all_space:
                    # Only include positions that don't conflict with fixed breakpoints
                    if not any(abs(pos - fixed) < 0.01 for fixed in self.fixed_breakpoints):
                        available_positions.append(pos)

                # Take the first n_variable_breakpoints positions
                variable_breakpoints = available_positions[:n_variable_breakpoints]
                params.extend(variable_breakpoints)

        # Parameters for each segment (except last if predefined)
        n_param_segments = n_segments - 1 if self.last_segment_params else n_segments

        for i in range(n_param_segments):
            # Use last segment parameters as base, adjust x0 for each segment
            if self.last_segment_params:
                a_init = self.last_segment_params['a']
                x0_init = self.last_segment_params['x0']
                n_init = self.last_segment_params['n']
            else:
                # Fallback to simple initial guesses
                a_init = np.mean(self.y_data) * 3
                x0_init = np.min(self.x_data)
                n_init = 1.7

            params.extend([a_init, x0_init, n_init])

        return np.array(params)

    def get_bounds(self, n_segments):
        """
        Define optimization bounds for all parameters.

        Establishes reasonable constraints for breakpoints and segment parameters
        to guide the optimization algorithm and prevent unrealistic solutions.

        Parameters
        ----------
        n_segments : int
            Number of segments to create bounds for

        Returns
        -------
        list of tuple
            Bounds for each parameter as (min, max) tuples
        """
        bounds = []

        if n_segments > 1:
            # Bounds for variable breakpoints only (fixed ones are not optimization variables)
            n_variable_breakpoints = n_segments - 1 - len(self.fixed_breakpoints)

            for i in range(n_variable_breakpoints):
                bounds.append((self.x_min, self.x_max))

        # Bounds for segment parameters
        n_param_segments = n_segments - 1 if self.last_segment_params else n_segments

        for _ in range(n_param_segments):
            # a: coefficient
            bounds.append((1, 1e4))
            # x0: offset in m
            bounds.append((self.x_min - abs(self.x_min*1.5), self.x_min + abs(self.x_min*1.5)))
            # n: power
            bounds.append((1.2, 5))

        return bounds
    
    def fit_segments(self, n_segments, maxiter=1000, popsize=100, **obj_kwargs):
        """
        Fit segmented rating curve using advanced differential evolution.

        This is the main optimization entry point that orchestrates the entire
        fitting process, from initial guesses through bounds checking to final
        segment creation and validation.

        Parameters
        ----------
        n_segments : int, range 1-5
            Number of segments to fit to the data
        maxiter : int, default 1000
            Maximum number of optimization iterations
        popsize : int, default 100
            Population size for differential evolution (affects search diversity)
        **obj_kwargs : dict
            Additional keyword arguments passed to create_objective_function()
            for customizing penalty weights and optimization behavior

        Returns
        -------
        dict
            Fitting results containing:
            - 'segments': List of fitted Segment objects
            - 'params': Raw optimization parameters
            - 'objective': Final objective function value
            - 'components': Detailed breakdown of penalty components
            - 'success': Boolean indicating optimization convergence
            - 'n_segments': Number of segments fitted

        Notes
        -----
        Automatically adds fitted segments to the RatingCurves instance for
        future curve crossing prevention.
        """
        objective = self.create_objective_function(**obj_kwargs)
        initial_guess = self.get_initial_guess(n_segments)
        bounds = self.get_bounds(n_segments)
        
        result = differential_evolution(
            lambda p: objective(p, n_segments),
            bounds,
            x0=initial_guess,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            atol=1e-10,
            tol=1e-10
        )
        
        segments = self.params_to_segments(result.x, n_segments)
        components = objective(result.x, n_segments, return_components=True)

        # Add the new fitted curve to the existing ones
        self.existing_curves.extend(segments)

        return {
            'segments': segments,
            'params': result.x,
            'objective': result.fun,
            'components': components,
            'success': result.success,
            'n_segments': n_segments
        }
    
    def plot_curves(self, station="", show_residuals=True, width=1000, height=700):
        """
        Create interactive multi-curve visualization using Plotly.

        Groups segments by curve ID and creates an interactive plot showing all curves
        with their residuals. Residuals are computed for each curve according to the
        date of validity and are color-coded by the curve they belong to.

        Parameters
        ----------
        station : str, optional
            Station identifier for plot title
        show_residuals : bool, default True
            Whether to include residual subplot below main plot
        width : int, default 1000
            Plot width in pixels
        height : int, default 700
            Plot height in pixels

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure with zoom, pan, and toggle capabilities

        Notes
        -----
        - Curves are automatically grouped by their curve ID (cid)
        - Each curve gets a unique color and legend entry
        - Residuals are calculated only for measurements within curve validity periods
        - Residuals are color-coded to match their corresponding curve
        - Reference lines show ±10% and ±20% error bounds
        """

        # Group segments by curve ID
        curves_by_cid = {}
        for seg in self.existing_curves:
            cid = seg.cid if seg.cid is not None else 'Default'
            if cid not in curves_by_cid:
                curves_by_cid[cid] = []
            curves_by_cid[cid].append(seg)

        # Create subplots if residuals requested and data available
        if show_residuals and self.x_data is not None and self.y_data is not None:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25],
                subplot_titles=['Rating Curves', ''],
                vertical_spacing=0.12,  # Increased spacing to prevent text overlap
                shared_xaxes=False  # Don't share x-axes to show ticks on both
            )
            main_row = 1
            residual_row = 2
        else:
            fig = go.Figure()
            main_row = None
            residual_row = None

        # Color palette for different curves
        color_palette = colors.qualitative.Set1

        # Generate smooth plotting domain
        x_plot = np.linspace(self.x_min, self.x_max, 1000)

        # Plot measurement data with date information in hover
        if self.x_data is not None and self.y_data is not None:
            scatter_trace = go.Scatter(
                x=self.x_data,
                y=self.y_data,
                mode='markers',
                name='Measurements',
                marker=dict(
                    color='black',
                    size=6,
                    opacity=0.7
                ),
                customdata=self.dates,
                hovertemplate='<b>Measurement</b><br>Date: %{customdata}<br>H: %{x:.2f} m<br>Q: %{y:.2f} m³/s<extra></extra>',
                showlegend=True
            )

            if main_row:
                fig.add_trace(scatter_trace, row=main_row, col=1)
            else:
                fig.add_trace(scatter_trace)

        # Plot each curve (grouped by cid)
        for curve_idx, (curve_id, segments) in enumerate(curves_by_cid.items()):
            curve_color = color_palette[curve_idx % len(color_palette)]

            # Plot curve segments
            for seg_idx, seg in enumerate(segments):
                # Create x range for this segment within expanded domain
                seg_x_start = seg.x_start
                seg_x_end = seg.x_end

                if seg_x_start < seg_x_end:
                    # Create fine-grained x values for this segment
                    mask = (x_plot >= seg_x_start) & (x_plot <= seg_x_end)
                    x_seg = x_plot[mask]

                    if len(x_seg) > 0:
                        y_seg = seg.evaluate(x_seg)

                        segment_trace = go.Scatter(
                            x=x_seg,
                            y=y_seg,
                            mode='lines',
                            name=curve_id,
                            line=dict(
                                color=curve_color,
                                width=3,
                                dash='solid' if seg_idx == 0 else 'dash'
                            ),
                            hovertemplate=f'<b>{curve_id}</b><br>H: %{{x:.2f}} m<br>Q: %{{y:.2f}} m³/s<extra></extra>',
                            legendgroup=curve_id,
                            showlegend=(seg_idx == 0)
                        )

                        if main_row:
                            fig.add_trace(segment_trace, row=main_row, col=1)
                        else:
                            fig.add_trace(segment_trace)

            # Calculate and plot residuals for valid measurements only
            if (show_residuals and main_row and
                self.x_data is not None and self.y_data is not None and self.dates is not None):

                # Parse curve validity dates
                try:
                    start_date, end_date = curve_id.split('_')
                except:
                    start_date, end_date = (None, None)

                if start_date is not None and end_date is not None:
                    
                    # Find measurements within curve validity period
                    valid_mask = np.array([
                        d is not None and start_date <= d < end_date
                        for d in self.dates
                    ])

                    if np.any(valid_mask):
                        # Get valid data points
                        x_valid = self.x_data[valid_mask]
                        y_valid = self.y_data[valid_mask]
                        dates_valid = self.dates[valid_mask]

                        # Calculate fitted values for valid points
                        y_fitted_valid = np.zeros_like(y_valid)
                        for i, x in enumerate(x_valid):
                            for seg in segments:
                                if seg.x_start <= x <= seg.x_end:
                                    y_fitted_valid[i] = seg.evaluate(x)
                                    break

                        # Calculate residuals
                        residuals_valid = 100 * (y_fitted_valid - y_valid) / y_valid

                        residual_trace = go.Scatter(
                            x=x_valid,
                            y=residuals_valid,
                            mode='markers',
                            name=f'{curve_id} - Residuals',
                            marker=dict(
                                color=curve_color,
                                size=5,
                                opacity=0.7
                            ),
                            customdata=dates_valid,
                            hovertemplate=f'<b>{curve_id} Residuals</b><br>Date: %{{customdata}}<br>H: %{{x:.2f}} m<br>Residual: %{{y:.1f}}%<extra></extra>',
                            legendgroup=curve_id,
                            showlegend=False
                        )

                        fig.add_trace(residual_trace, row=residual_row, col=1)

                else:
                    # Fallback for curves without valid date format - use all data
                    y_fitted = np.zeros_like(self.y_data)
                    for i, x in enumerate(self.x_data):
                        for seg in segments:
                            if seg.x_start <= x <= seg.x_end:
                                y_fitted[i] = seg.evaluate(x)
                                break

                    residuals = 100 * (y_fitted - self.y_data) / self.y_data

                    residual_trace = go.Scatter(
                        x=self.x_data,
                        y=residuals,
                        mode='markers',
                        name=f'{curve_id} - Residuals',
                        marker=dict(
                            color=curve_color,
                            size=5,
                            opacity=0.7
                        ),
                        customdata=self.dates,
                        hovertemplate=f'<b>{curve_id} Residuals</b><br>Date: %{{customdata}}<br>H: %{{x:.2f}} m<br>Residual: %{{y:.1f}}%<extra></extra>',
                        legendgroup=curve_id,
                        showlegend=False
                    )

                    fig.add_trace(residual_trace, row=residual_row, col=1)

        # Add reference lines for residuals
        if show_residuals and main_row:
            reference_lines = [
                (0, 'Zero', 'black', 'solid'),
                (10, '+10%', 'red', 'dash'),
                (-10, '-10%', 'red', 'dash'),
                (20, '+20%', 'orange', 'dash'),
                (-20, '-20%', 'orange', 'dash')
            ]

            for y_val, name, color, dash in reference_lines:
                ref_trace = go.Scatter(
                    x=[self.x_min, self.x_max],
                    y=[y_val, y_val],
                    mode='lines',
                    line=dict(color=color, dash=dash, width=1),
                    name=name if y_val == 0 else None,
                    showlegend=(y_val == 0),
                    hoverinfo='skip'
                )
                fig.add_trace(ref_trace, row=residual_row, col=1)

        # Update layout
        title = f'Rating Curves - Station {station}' if station else 'Rating Curves'

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            width=width,
            height=height,
            hovermode='closest',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        # Update axes
        if main_row:
            fig.update_xaxes(title_text="Water Level (m)", row=main_row, col=1)
            fig.update_yaxes(title_text="Discharge (m³/s)", type="log", row=main_row, col=1)

            if show_residuals:
                fig.update_xaxes(title_text="Water Level (m)", row=residual_row, col=1)
                fig.update_yaxes(title_text="Residual (%)", row=residual_row, col=1)
        else:
            fig.update_xaxes(title_text="Water Level (m)")
            fig.update_yaxes(title_text="Discharge (m³/s)", type="log")

        return fig

    def plot_results(self, cid, station="", show_components=True, figsize=(12, 8)):
        """
        Visualize a single rating curve with comprehensive data analysis.

        Parameters
        ----------
        cid : str
            Curve ID to plot (specific curve to analyze)
        station : str, optional
            Station identifier for title
        show_components : bool, default True
            Whether to show residual plot as subplot
        figsize : tuple, default (12, 8)
            Figure size (width, height) in inches
        """
        # Get segments for the specified curve ID
        segments = [seg for seg in self.existing_curves if seg.cid == cid]

        if not segments:
            print(f"No segments found for curve ID: {cid}")
            available_cids = list(set(seg.cid for seg in self.existing_curves))
            print(f"Available curve IDs: {available_cids}")
            return None

        # Create subplots
        if show_components:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = None

        # Plot observed data
        ax1.scatter(self.x_data, self.y_data, alpha=0.6, color='black', s=30,
                   label='Observed', zorder=3)

        # Generate smooth curve for plotting fitted segments
        x_plot = np.linspace(self.x_min, self.x_max, 500)

        # Plot each segment with different colors
        colors = plt.cm.Set1(np.linspace(0, 1, len(segments)))

        for i, seg in enumerate(segments):
            # Create x range for this segment
            mask = (x_plot >= seg.x_start) & (x_plot <= seg.x_end)
            x_seg = x_plot[mask]

            if len(x_seg) > 0:
                y_seg = seg.evaluate(x_seg)

                ax1.plot(x_seg, y_seg, color=colors[i], linewidth=2,
                        label=f"Segment {i+1}: Q = {seg.a:.2f}×(H-{seg.x0:.2f})^{seg.n:.2f}")

                # Add vertical lines at segment boundaries
                if i < len(segments) - 1:
                    boundary_x = seg.x_end
                    ax1.axvline(boundary_x, color=colors[i], linestyle='--', alpha=0.7)

        # Set main plot properties
        ax1.set_xlabel('Water Level (m)')
        ax1.set_ylabel('Discharge (m³/s)')
        ax1.set_title(f'Rating Curve Analysis - Station {station}\n'
                      f'Curve ID: {cid}\n'
                      f'{len(segments)} segments')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)

        # Log scale for better visualization
        ax1.set_yscale('log')

        if show_components and ax2 is not None:
            # Calculate residuals for this specific curve
            y_fitted = np.zeros_like(self.y_data)
            for i, x in enumerate(self.x_data):
                for seg in segments:
                    if seg.x_start <= x <= seg.x_end:
                        y_fitted[i] = seg.evaluate(x)
                        break

            residuals = 100 * (y_fitted - self.y_data) / self.y_data

            # Calculate comprehensive metrics
            mape = np.mean(np.abs(residuals))  # Mean Percentage Error
            bias = np.mean(residuals)  # Bias (mean residual)
            positive_errors = np.sum(residuals > 0) / len(residuals) * 100  # % positive errors
            negative_errors = np.sum(residuals < 0) / len(residuals) * 100  # % negative errors

            # Additional statistics
            within_10pct = np.sum(np.abs(residuals) <= 10) / len(residuals) * 100
            within_20pct = np.sum(np.abs(residuals) <= 20) / len(residuals) * 100
            max_error = np.max(np.abs(residuals))

            # Plot residuals
            ax2.scatter(self.x_data, residuals, alpha=0.6, color='blue', s=20)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='±10%')
            ax2.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='±20%')
            ax2.axhline(y=-20, color='orange', linestyle='--', alpha=0.5)

            # Enhanced metrics text box
            metrics_text = (f'MAPE: {mape:.1f}%\n'
                          f'Bias: {bias:.1f}%\n'
                          f'Max Error: {max_error:.1f}%\n'
                          f'% Positives: {positive_errors:.1f}%\n'
                          f'% Negatives: {negative_errors:.1f}%\n'
                          f'Within ±10%: {within_10pct:.1f}%\n'
                          f'Within ±20%: {within_20pct:.1f}%')

            ax1.text(0.98, 0.02, metrics_text,
                    transform=ax1.transAxes, fontsize=9,
                    va='bottom', ha='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax2.set_xlabel('Water Level (m)')
            ax2.set_ylabel('Residual (%)')
            ax2.set_title('Residuals (Fitted - Observed)/Observed × 100%')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')

            # Set reasonable y-limits for residuals
            residual_max = np.max(np.abs(residuals))
            ax2.set_ylim(-min(50, residual_max * 1.2), min(50, residual_max * 1.2))

        plt.tight_layout()
        plt.show()

        if show_components:
            return fig, (ax1, ax2)
        else:
            return fig, ax1
