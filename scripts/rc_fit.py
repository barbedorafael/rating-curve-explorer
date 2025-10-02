import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import huber
import matplotlib.pyplot as plt
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as colors

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


class RatigCurveFitter:
    def __init__(self, x_data, y_data,
                 last_segment_params=None, existing_curves=None, fixed_breakpoints=None,
                 x_min=None, x_max=None,):
        """
        Initialize the fitter

        Parameters:
        -----------
        x_data, y_data: array-like
            Data points to fit
        last_segment_params: dict or None
            Parameters for the last segment if predefined
            Should contain: {'a': float, 'x0': float, 'n': float, 'x_start': float}
        existing_curves: list of Segment objects, optional
            Existing curves that the new curve should not cross
        fixed_breakpoints: list of float, optional
            Fixed breakpoint positions that should not be optimized
        """
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.x_min = x_min
        self.x_max = x_max
        if not self.x_min:
            self.x_min = self.x_data.min()
        if not self.x_max:
            self.x_max = self.x_data.max()
        self.last_segment_params = last_segment_params
        self.existing_curves = existing_curves or []
        self.fixed_breakpoints = fixed_breakpoints or []

        # Sort data by x
        idx = np.argsort(self.x_data)
        self.x_data = self.x_data[idx]
        self.y_data = self.y_data[idx]


        
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
                                  curve_crossing_weight=20):
        """
        Create a custom objective function with various criteria

        Parameters:
        -----------
        loss_weight: float
            Weight for loss function penalty (MPE)
        continuity_weight: float
            Weight for continuity violation penalty
        min_segment_range: float or None
            Minimum x-range each segment must cover (prevents tiny segments)
        min_range_weight: float
            Weight for penalty when segment x-range is too small
        min_points_per_segment: int
            Minimum number of data points each segment must cover
        min_points_weight: float
            Weight for penalty when segments have too few points
        param_deviation_weight: float
            Weight for penalty when segment parameters deviate from reference
        n_deviation_weight: float
            Additional weight specifically for 'n' parameter deviations
        bias_weight: float
            Weight for bias penalty to balance errors across x-values
        n_bias_bins: int
            Number of bins for bias penalty calculation
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
        """Convert parameter array to segment objects"""
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
        """Generate initial guess for parameters"""
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
        """Get parameter bounds for optimization"""
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
        Fit curve with fixed number of segments
        
        Parameters:
        -----------
        n_segments: int
            Number of segments (1 to 5)
        maxiter: int
            differential evolution argument
        popsize: int
            differential evolution argument
        obj_kwargs: dict
            Arguments for objective function creation
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
        self.existing_curves.extend(segments)

        return {
            'segments': segments,
            'params': result.x,
            'objective': result.fun,
            'components': components,
            'success': result.success,
            'n_segments': n_segments
        }
    
    def load_existing_segments(self, df):
        """
        Load rating curve data and convert to Segment objects
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

    def plot_results(self, result, station="", show_components=True, figsize=(12, 8)):
        """
        Visualize the fitted curve and data with enhanced plotting style.

        Parameters
        ----------
        result : dict
            Result from fit_segments containing segments and fitting info
        show_components : bool, default True
            Whether to show residual plot as subplot
        figsize : tuple, default (12, 8)
            Figure size (width, height) in inches
        """
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
        segments = result['segments']

        # Plot each segment with different colors
        colors = plt.cm.Set1(np.linspace(0, 1, len(segments)))

        for i, seg in enumerate(segments):
            # Create x range for this segment
            mask = (x_plot >= seg.x_start) & (x_plot <= seg.x_end)
            x_seg = x_plot[mask]

            if len(x_seg) > 0:
                y_seg = seg.evaluate(x_seg)

                ax1.plot(x_seg, y_seg, color=colors[i], linewidth=2,
                        label=f"Segment {i+1}: Q = {seg.a:.2f}×(X-{seg.x0:.2f})^{seg.n:.2f}")

                # Add vertical lines at segment boundaries
                if i < len(segments) - 1:
                    boundary_x = seg.x_end
                    ax1.axvline(boundary_x, color=colors[i], linestyle='--', alpha=0.7)

        # Set main plot properties
        ax1.set_xlabel('Water Level')
        ax1.set_ylabel('Discharge')
        ax1.set_title(f'Station number: {station}\n'
                      f'Segmented Power Curve Fit\n'
                      f'{len(segments)} segments')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)

        # Log scale for better visualization
        ax1.set_yscale('log')

        if show_components and ax2 is not None:
            # Calculate residuals
            y_fitted = np.zeros_like(self.y_data)
            for i, x in enumerate(self.x_data):
                for seg in segments:
                    if seg.x_start <= x <= seg.x_end:
                        y_fitted[i] = seg.evaluate(x)
                        break

            residuals = 100 * (y_fitted - self.y_data) / self.y_data

            # Calculate metrics
            mpe = np.mean(np.abs(residuals))  # Mean Percentage Error
            bias = np.mean(residuals)  # Bias (mean residual)
            positive_errors = np.sum(residuals > 0) / len(residuals) * 100  # % positive errors
            negative_errors = np.sum(residuals < 0) / len(residuals) * 100  # % negative errors

            # Plot residuals
            ax2.scatter(self.x_data, residuals, alpha=0.6, color='blue', s=20)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='±10%')
            ax2.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='±20%')
            ax2.axhline(y=-20, color='orange', linestyle='--', alpha=0.5)

            # Add metrics text box
            metrics_text = f'MPE: {mpe:.1f}%\nBias: {bias:.1f}%\n+Errors: {positive_errors:.1f}%\n-Errors: {negative_errors:.1f}%'
            ax1.text(0.98, 0.02, metrics_text,
                    transform=ax1.transAxes, fontsize=10,
                    va='bottom', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax2.set_xlabel('Water Level')
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

    def plot_curves(self, station="", show_residuals=True, width=1000, height=700):
        """
        Plot multiple rating curves using Plotly with interactive features.

        Parameters
        ----------
        station : str, optional
            Station identifier for title
        show_residuals : bool, default True
            Whether to show residual subplot
        domain_expansion : float, default 0.5
            Factor to expand plotting domain beyond measurements (0.5 = 50% each side)
        width : int, default 1000
            Plot width in pixels
        height : int, default 700
            Plot height in pixels

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure
        """

        # Group segments by curve ID
        curves_by_cid = {}
        for seg in self.existing_curves:
            cid = seg.cid if seg.cid is not None else 'Default'
            if cid not in curves_by_cid:
                curves_by_cid[cid] = []
            curves_by_cid[cid].append(seg)
        
        print(curves_by_cid)

        # Create subplots if residuals requested and data available
        if show_residuals and self.x_data is not None and self.y_data is not None:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25],
                subplot_titles=['Rating Curves', ''],
                vertical_spacing=0.09,  # Increased spacing to prevent text overlap
                shared_xaxes=True
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

        # Plot measurement data once (shared across all curves)
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

                        # Create segment name and info
                        segment_name = f'{curve_id}'
                        equation_text = f'Q = {seg.a:.3f}×(H-{seg.x0:.2f})^{seg.n:.3f}'

                        segment_trace = go.Scatter(
                            x=x_seg,
                            y=y_seg,
                            mode='lines',
                            name=segment_name,
                            line=dict(
                                color=curve_color,
                                width=3,
                                dash='solid' if seg_idx == 0 else 'dash'
                            ),
                            hovertemplate=f'<b>{curve_id}</b><br>H: %{{x:.2f}}<br>Q: %{{y:.2f}}<extra></extra>',
                            legendgroup=curve_id,
                            showlegend=(seg_idx == 0)  # Only show legend for first segment
                        )

                        if main_row:
                            fig.add_trace(segment_trace, row=main_row, col=1)
                        else:
                            fig.add_trace(segment_trace)

            # Calculate and plot residuals if data available
            if (show_residuals and main_row and
                self.x_data is not None and self.y_data is not None):

                # Calculate fitted values for this curve
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