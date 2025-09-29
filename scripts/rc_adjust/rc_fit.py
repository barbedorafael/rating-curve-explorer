import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares
from scipy.stats import huber
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import warnings

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
    
    def evaluate_safe(self, x):
        """Evaluate with handling for negative base with non-integer power"""
        base = x - self.x0
        if np.any(base <= 0) and not float(self.n).is_integer():
            # Use complex arithmetic and take real part
            return np.real(self.a * np.power(base + 0j, self.n))
        return self.a * np.power(base, self.n)

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
                                  continuity_weight=100,
                                  max_error_threshold=0.15,
                                  max_error_weight=10,
                                  target_range=(0.3, 0.7),
                                  range_weight=1,
                                  outlier_method='huber',
                                  outlier_threshold=1.0):
        """
        Create a custom objective function with various criteria
        
        Parameters:
        -----------
        continuity_weight: float
            Weight for continuity violation penalty
        max_error_threshold: float
            Maximum allowed relative error (e.g., 0.15 for 15%)
        max_error_weight: float
            Weight for maximum error violation
        target_range: tuple
            Target range for specific x values (as fraction of x range)
        range_weight: float
            Weight for target range penalty
        outlier_method: str
            Method for handling outliers ('huber', 'soft_l1', 'cauchy')
        outlier_threshold: float
            Threshold for outlier detection
        """
        
        def objective(params, n_segments, return_components=False):
            segments = self.params_to_segments(params, n_segments)
            
            # Compute fitted values
            y_fitted = np.zeros_like(self.y_data)
            for i, x in enumerate(self.x_data):
                for seg in segments:
                    if seg.x_start <= x <= seg.x_end:
                        y_fitted[i] = seg.evaluate_safe(x)
                        break
            
            # 1. Base fitting error with outlier handling
            residuals = self.y_data - y_fitted
            rel_errors = np.abs(residuals) / (np.abs(self.y_data) + 1e-10)
            
            if outlier_method == 'huber':
                base_loss = huber.huber(outlier_threshold, residuals).sum()
            elif outlier_method == 'soft_l1':
                base_loss = np.sum(2 * (np.sqrt(1 + (residuals/outlier_threshold)**2) - 1))
            elif outlier_method == 'cauchy':
                base_loss = np.sum(outlier_threshold**2 * np.log(1 + (residuals/outlier_threshold)**2))
            else:
                base_loss = np.sum(residuals**2)
            
            # 2. Continuity penalty
            continuity_penalty = 0
            for i in range(len(segments) - 1):
                x_break = segments[i].x_end
                val1 = segments[i].evaluate_safe(x_break)
                val2 = segments[i+1].evaluate_safe(x_break)
                rel_diff = abs(val1 - val2) / (abs(val1) + 1e-10)
                if rel_diff > 0.05:  # 5% threshold
                    continuity_penalty += (rel_diff - 0.05)**2
            
            # 3. Maximum error penalty
            max_error_penalty = 0
            max_rel_error = np.max(rel_errors)
            if max_rel_error > max_error_threshold:
                max_error_penalty = (max_rel_error - max_error_threshold)**2
            
            # 4. Target range penalty (example: penalize if certain range not well fitted)
            range_penalty = 0
            if target_range:
                x_range_min = self.x_min + target_range[0] * (self.x_max - self.x_min)
                x_range_max = self.x_min + target_range[1] * (self.x_max - self.x_min)
                mask = (self.x_data >= x_range_min) & (self.x_data <= x_range_max)
                if np.any(mask):
                    range_errors = rel_errors[mask]
                    range_penalty = np.mean(range_errors**2)
            
            # Total objective
            total = (base_loss + 
                    continuity_weight * continuity_penalty +
                    max_error_weight * max_error_penalty +
                    range_weight * range_penalty)
            
            if return_components:
                return {
                    'total': total,
                    'base_loss': base_loss,
                    'continuity_penalty': continuity_penalty,
                    'max_error_penalty': max_error_penalty,
                    'range_penalty': range_penalty
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
            n = 1.0
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
        
        for i in range(n_param_segments):
            # a: coefficient
            bounds.append((None, None))
            # x0: offset
            bounds.append((None, None))
            # n: power (typically positive)
            bounds.append((0.1, 10))
        
        return bounds
    
    def fit_fixed_segments(self, n_segments, method='differential_evolution', **obj_kwargs):
        """
        Fit curve with fixed number of segments
        
        Parameters:
        -----------
        n_segments: int
            Number of segments (1 to 5)
        method: str
            Optimization method ('differential_evolution', 'L-BFGS-B', 'trust-constr')
        obj_kwargs: dict
            Arguments for objective function creation
        """
        objective = self.create_objective_function(**obj_kwargs)
        initial_guess = self.get_initial_guess(n_segments)
        bounds = self.get_bounds(n_segments)
        
        if method == 'differential_evolution':
            result = differential_evolution(
                lambda p: objective(p, n_segments),
                bounds,
                seed=42,
                maxiter=1000,
                popsize=15,
                atol=1e-10,
                tol=1e-10
            )
        elif method == 'L-BFGS-B':
            result = minimize(
                lambda p: objective(p, n_segments),
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds
            )
        elif method == 'trust-constr':
            result = minimize(
                lambda p: objective(p, n_segments),
                initial_guess,
                method='trust-constr',
                bounds=bounds
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
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
    
    def optimize_segments(self, max_segments=5, method='differential_evolution', 
                         use_bic=True, **obj_kwargs):
        """
        Optimize both number of segments and their parameters
        
        Parameters:
        -----------
        max_segments: int
            Maximum number of segments to consider
        method: str
            Optimization method
        use_bic: bool
            Use BIC criterion for model selection
        """
        results = []
        
        for n_seg in range(1, max_segments + 1):
            print(f"Fitting with {n_seg} segments...")
            try:
                result = self.fit_fixed_segments(n_seg, method, **obj_kwargs)
                
                # Calculate fit quality metrics
                segments = result['segments']
                y_fitted = np.zeros_like(self.y_data)
                for i, x in enumerate(self.x_data):
                    for seg in segments:
                        if seg.x_start <= x <= seg.x_end:
                            y_fitted[i] = seg.evaluate_safe(x)
                            break
                
                residuals = self.y_data - y_fitted
                sse = np.sum(residuals**2)
                n_params = len(result['params'])
                n_data = len(self.x_data)
                
                # BIC for model selection
                if use_bic:
                    bic = n_data * np.log(sse / n_data) + n_params * np.log(n_data)
                    result['bic'] = bic
                
                result['sse'] = sse
                result['rmse'] = np.sqrt(sse / n_data)
                results.append(result)
                
            except Exception as e:
                print(f"Failed for {n_seg} segments: {e}")
                continue
        
        # Select best model
        if use_bic:
            best_idx = np.argmin([r['bic'] for r in results])
        else:
            best_idx = np.argmin([r['objective'] for r in results])
        
        return results[best_idx], results
    
    def plot_results(self, result, show_components=True):
        """Visualize the fitted curve and data"""
        fig, axes = plt.subplots(2 if show_components else 1, 1, 
                                 figsize=(10, 8 if show_components else 4))
        
        if not show_components:
            axes = [axes]
        
        # Main plot
        ax = axes[0]
        ax.scatter(self.x_data, self.y_data, alpha=0.6, label='Data')
        
        # Plot fitted curve
        x_plot = np.linspace(self.x_min, self.x_max, 500)
        y_plot = np.zeros_like(x_plot)
        
        segments = result['segments']
        colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
        
        for seg, color in zip(segments, colors):
            mask = (x_plot >= seg.x_start) & (x_plot <= seg.x_end)
            x_seg = x_plot[mask]
            if len(x_seg) > 0:
                y_seg = seg.evaluate_safe(x_seg)
                y_plot[mask] = y_seg
                ax.plot(x_seg, y_seg, color=color, linewidth=2,
                       label=f'Seg {segments.index(seg)+1}: n={seg.n:.2f}')
        
        # Mark breakpoints
        for i in range(len(segments) - 1):
            x_break = segments[i].x_end
            ax.axvline(x_break, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Segmented Power Curve Fit ({len(segments)} segments)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Component plot
        if show_components and len(axes) > 1:
            ax2 = axes[1]
            
            # Calculate residuals
            y_fitted = np.zeros_like(self.y_data)
            for i, x in enumerate(self.x_data):
                for seg in segments:
                    if seg.x_start <= x <= seg.x_end:
                        y_fitted[i] = seg.evaluate_safe(x)
                        break
            
            residuals = self.y_data - y_fitted
            rel_errors = np.abs(residuals) / (np.abs(self.y_data) + 1e-10)
            
            ax2.scatter(self.x_data, rel_errors * 100, alpha=0.6)
            ax2.axhline(15, color='red', linestyle='--', label='15% threshold')
            ax2.axhline(5, color='orange', linestyle='--', label='5% threshold')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Relative Error (%)')
            ax2.set_title('Fitting Errors')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
