import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import huber
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Segment:
    """Represents a power function segment"""
    a: float
    x0: float
    n: float
    x_start: float
    x_end: float
    
    def evaluate(self, x):
        """Evaluate the power function at given x values"""
        return self.a * (x - self.x0) ** self.n

class SegmentedPowerCurveFitter:
    def __init__(self, x_data, y_data, last_segment_params=None):
        """
        Initialize the fitter
        
        Parameters:
        -----------
        x_data, y_data: array-like
            Data points to fit
        last_segment_params: dict or None
            Parameters for the last segment if predefined
            Should contain: {'a': float, 'x0': float, 'n': float, 'x_start': float}
        """
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.last_segment_params = last_segment_params
        
        # Sort data by x
        idx = np.argsort(self.x_data)
        self.x_data = self.x_data[idx]
        self.y_data = self.y_data[idx]
        
        self.x_min = self.x_data.min()
        self.x_max = self.x_data.max()
        
    def create_objective_function(self, 
                                  loss_weight=10,
                                  continuity_threshold=0.05,
                                  continuity_weight=10,
                                  min_segment_range=0.1,
                                  min_range_weight=1,
                                  min_points_per_segment=2,
                                  min_points_weight=1):
        """
        Create a custom objective function with various criteria
        
        Parameters:
        -----------
        loss_weight: float
            Weight for loss function penalty (MPE)
                continuity_weight: float
            Weight for continuity violation penalty
        outlier_method: str
            Method for handling outliers ('huber', 'soft_l1', 'cauchy')
        outlier_threshold: float
            Threshold for outlier detection
        min_segment_range: float or None
            Minimum x-range each segment must cover (prevents tiny segments)
        min_range_weight: float
            Weight for penalty when segment x-range is too small
        min_points_per_segment: int
            Minimum number of data points each segment must cover
        min_points_weight: float
            Weight for penalty when segments have too few points
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
            
            # Total objective
            total = (loss_weight * loss_penalty + 
                    continuity_weight * continuity_penalty +
                    min_range_weight * min_range_penalty +
                    min_points_weight * min_points_penalty)
            
            if return_components:
                return {
                    'total': total,
                    'base_loss': loss_penalty,
                    'continuity_penalty': continuity_penalty,
                    'min_range_penalty': min_range_penalty,
                    'min_points_penalty': min_points_penalty,
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
            # Extract breakpoints (n_segments - 1 breakpoints)
            breakpoints = sorted(params[:n_segments-1])
            
            # Add boundaries
            all_breaks = [self.x_min] + list(breakpoints) + [self.x_max]
            
            # Extract segment parameters
            param_idx = n_segments - 1
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
            # Initial breakpoints - evenly spaced
            breakpoints = np.linspace(self.x_min, self.x_max, n_segments + 1)[1:-1]
            params.extend(breakpoints)
        
        # Parameters for each segment (except last if predefined)
        n_param_segments = n_segments - 1 if self.last_segment_params else n_segments
        
        for i in range(n_param_segments):
            # Simple initial guesses
            a = np.mean(self.y_data)
            x0 = self.x_min
            n = 1.5
            params.extend([a, x0, n])
        
        return np.array(params)
    
    def get_bounds(self, n_segments):
        """Get parameter bounds for optimization"""
        bounds = []
        
        if n_segments > 1:
            # Bounds for breakpoints
            for i in range(n_segments - 1):
                bounds.append((self.x_min, self.x_max))
        
        # Bounds for segment parameters
        n_param_segments = n_segments - 1 if self.last_segment_params else n_segments
        
        for _ in range(n_param_segments):
            # a: coefficient
            bounds.append((1, 1e4))
            # x0: offset
            bounds.append((-100, self.x_min*0.9))
            # n: power
            bounds.append((1.2, 5))
        
        return bounds
    
    def fit_segments(self, n_segments, **obj_kwargs):
        """
        Fit curve with fixed number of segments
        
        Parameters:
        -----------
        n_segments: int
            Number of segments (1 to 5)
        obj_kwargs: dict
            Arguments for objective function creation
        """
        objective = self.create_objective_function(**obj_kwargs)
        initial_guess = self.get_initial_guess(n_segments)
        bounds = self.get_bounds(n_segments)
        
        result = differential_evolution(
            lambda p: objective(p, n_segments),
            bounds,
            seed=42,
            maxiter=1000,
            popsize=50,
            atol=1e-10,
            tol=1e-10
        )
        
        segments = self.params_to_segments(result.x, n_segments)
        components = objective(result.x, n_segments, return_components=True)
        
        return {
            'segments': segments,
            'params': result.x,
            'objective': result.fun,
            'components': components,
            'success': result.success,
            'n_segments': n_segments
        }
    
    def plot_results(self, result, show_components=True, figsize=(12, 8)):
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
        ax1.set_title(f'Segmented Power Curve Fit\n'
                     f'{len(segments)} segments - Objective: {result["objective"]:.2f}')
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

            # Plot residuals
            ax2.scatter(self.x_data, residuals, alpha=0.6, color='blue', s=20)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='±10%')
            ax2.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='±20%')
            ax2.axhline(y=-20, color='orange', linestyle='--', alpha=0.5)

            ax2.set_xlabel('Water Level')
            ax2.set_ylabel('Residual (%)')
            ax2.set_title('Residuals (Fitted - Observed)/Observed × 100%')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')

            # Set reasonable y-limits for residuals
            residual_max = np.max(np.abs(residuals))
            ax2.set_ylim(-min(50, residual_max * 1.2), min(50, residual_max * 1.2))

        plt.tight_layout()

        if show_components:
            return fig, (ax1, ax2)
        else:
            return fig, ax1
