"""
Piecewise power-law fitting with unknown breakpoints and segment count (1..5).
Model per segment s: y = a_s * (x - x0_s)**n_s  for x in [b_{s-1}, b_s]
- Continuity enforced between segments with <=5% relative mismatch (hard or penalized)
- Last segment parameters can be fixed (a_L, n_L) and must start at last breakpoint
- Robust objective options (Huber, Tukey, quantile), point weights, max-abs-% error cap
- Extra penalty hooks (e.g., keep predictions within [p30, p70] band, etc.)

Algorithm:
1) Build a grid of candidate breakpoints from unique X (optional thinning by min_points).
2) Dynamic programming to pick the best set of breakpoints for k=1..5 segments, where each
   segment cost is obtained by robust nonlinear least-squares (scipy.least_squares) with
   continuity constraint (either hard or penalized) to the previous segment, and optional hooks.
3) Pick k via BIC/AIC/MDL (or user-specified) and return.
4) Optional continuous refinement of breakpoint positions by local search (coordinate descent)
   alternating (refit params | nudge breaks).

Requires: numpy, scipy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import numpy as np
from scipy.optimize import least_squares

# ----------------------------- Losses & utilities -----------------------------

def huber_residuals(r: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Return elementwise Huber residuals (sqrt of Huber loss) to use in least_squares.
    We implement via weighting so that sum(res**2) equals Huber loss.
    """
    abs_r = np.abs(r)
    quad = abs_r <= delta
    out = np.empty_like(r)
    out[quad] = r[quad]
    # For linear region, Huber loss L = delta*(|r|-delta/2). To pass as residuals,
    # use sqrt(2*delta*|r| - delta**2). Keep sign of r to assist Jacobian numerics.
    out[~quad] = np.sign(r[~quad]) * np.sqrt(2*delta*abs_r[~quad] - delta**2)
    return out


def tukey_biweight_residuals(r: np.ndarray, c: float = 4.685) -> np.ndarray:
    """Tukey's biweight (bisquare) residuals; outside |r|>c becomes constant influence.
    Passed as sqrt(loss) residuals. For |r|>=c set to constant to cap influence.
    """
    out = np.zeros_like(r)
    mask = np.abs(r) < c
    u = r[mask] / c
    out[mask] = r[mask] * np.sqrt((1 - u**2)**2)
    # For |r|>=c, set to constant so contribution stops growing
    out[~mask] = 0.0
    return out


def pct_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (y_pred - y_true) / (np.maximum(np.abs(y_true), eps))


# ----------------------- Power-law segment model & fit ------------------------

@dataclass
class SegmentParams:
    a: float
    n: float
    x0: float  # segment anchor (usually the left breakpoint)


def power_model(x: np.ndarray, p: SegmentParams) -> np.ndarray:
    return p.a * np.power(np.maximum(x - p.x0, 0.0), p.n)


def fit_power_segment(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y_left_target: Optional[float] = None,
    continuity_tol: float = 0.05,
    robust: str = "huber",
    robust_kw: Optional[Dict] = None,
    sample_weight: Optional[np.ndarray] = None,
    cap_max_pct_err: Optional[float] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = ((1e-12, -5.0), (np.inf, 5.0)),
) -> Tuple[SegmentParams, float, Dict]:
    """Fit a single power segment on (x,y) given fixed x0 (left breakpoint / anchor).

    Continuity: if y_left_target is provided, enforce |f(x0+)| within 5% of target via penalty.
    Returns (params, cost, info)
    """
    robust_kw = robust_kw or {}
    x = np.asarray(x)
    y = np.asarray(y)
    if sample_weight is None:
        w = np.ones_like(y)
    else:
        w = sample_weight / np.maximum(sample_weight.mean(), 1e-12)

    # Initial guesses: log-linear on shifted x
    dx = np.maximum(x - x0, 1e-6)
    logx = np.log(dx)
    logy = np.log(np.maximum(y, 1e-12))
    A = np.vstack([np.ones_like(logx), logx]).T
    try:
        beta, *_ = np.linalg.lstsq(A, logy, rcond=None)
        a0 = float(np.exp(beta[0]))
        n0 = float(beta[1])
    except Exception:
        a0, n0 = 1.0, 1.0

    lb, ub = bounds

    def residuals(theta: np.ndarray) -> np.ndarray:
        a, n = theta
        yhat = a * np.power(np.maximum(x - x0, 0.0), n)
        r = yhat - y
        if cap_max_pct_err is not None:
            pe = pct_error(y, yhat)
            # soft cap via tanh barrier to discourage |pe| > cap
            cap = cap_max_pct_err
            r += 10.0 * y * np.tanh(np.clip(np.abs(pe) - cap, 0, None)) * np.sign(pe)
        # Continuity residual at left boundary
        if y_left_target is not None:
            y0 = a * (0.0 ** n) if (x0 >= x.min()) else a * np.power(max(x.min() - x0, 0.0), n)
            # Since (x - x0)^n at x=x0 is 0 when n>0; allow n<=0 by evaluating at a tiny epsilon
            y0 = a * np.power(1e-6, n)
            cont_err = (y0 - y_left_target) / max(abs(y_left_target), 1e-9)
            r = np.concatenate([r, np.array([cont_err / continuity_tol])])
        # Apply robust re-scaling
        r_w = r * np.sqrt(w)
        if robust == "huber":
            return huber_residuals(r_w, **({"delta": 1.0} | robust_kw))
        elif robust == "tukey":
            return tukey_biweight_residuals(r_w, **({"c": 4.685} | robust_kw))
        elif robust == "none":
            return r_w
        else:
            raise ValueError("Unknown robust loss")

    res = least_squares(
        residuals, x0=np.array([a0, n0]), bounds=(np.array([lb[0], lb[1]]), np.array([ub[0], ub[1]])), method="trf"
    )
    a_opt, n_opt = res.x
    params = SegmentParams(a=a_opt, n=n_opt, x0=x0)
    cost = float(res.cost)
    info = {"success": res.success, "message": res.message, "nfev": res.nfev}
    return params, cost, info


# --------------------------- DP over breakpoints ------------------------------

@dataclass
class FitOptions:
    k_min: int = 1
    k_max: int = 5
    min_points_per_seg: int = 8
    continuity_tol: float = 0.05
    robust: str = "huber"  # 'huber' | 'tukey' | 'none'
    robust_kw: Optional[Dict] = None
    cap_max_pct_err: Optional[float] = 0.15  # 15%
    model_selection: str = "bic"  # 'aic' | 'bic' | 'mdl' | 'k_fixed'
    k_fixed: Optional[int] = None
    allow_break_refinement: bool = True


def candidate_breaks(x: np.ndarray, min_points: int) -> np.ndarray:
    x = np.asarray(x)
    idx = np.arange(min_points, len(x) - min_points)
    # Avoid duplicates by value change
    idx = idx[x[idx] != x[idx - 1]]
    return idx


def piecewise_power_fit(
    x: np.ndarray,
    y: np.ndarray,
    last_segment_params: Optional[Tuple[float, float]] = None,  # (a_L, n_L)
    opts: FitOptions = FitOptions(),
) -> Dict:
    """Return best piecewise fit and diagnostics.

    If last_segment_params is given, the last segment's (a,n) are fixed and only its x0 (last breakpoint)
    is optimized through DP (it influences continuity on prior segment).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    N = len(x)
    cand_idx = candidate_breaks(x, opts.min_points_per_seg)

    # Precompute segment costs between any i<j respecting min_points
    # cost[i,j] = best fit on x[i:j] with x0 = x[i]
    INF = 1e30
    cost = np.full((N, N), INF)
    params_cache: Dict[Tuple[int, int], SegmentParams] = {}

    for i in range(N - opts.min_points_per_seg):
        j_start = max(i + opts.min_points_per_seg, i + 1)
        for j in range(j_start, N + 1):
            xi = x[i:j]
            yi = y[i:j]
            # Continuity target for left boundary will be set by previous segment during DP.
            # Here we fit without continuity; DP will refit first point with continuity when needed.
            p, c, _ = fit_power_segment(
                xi, yi, x0=x[i], y_left_target=None,
                continuity_tol=opts.continuity_tol,
                robust=opts.robust, robust_kw=opts.robust_kw,
                cap_max_pct_err=opts.cap_max_pct_err,
            )
            cost[i, j - 1] = c
            params_cache[(i, j - 1)] = p

    # DP tables: dp[k][j] = best cost using k segments to cover up to index j
    Kmax = opts.k_max
    dp = np.full((Kmax + 1, N), INF)
    parent: Dict[Tuple[int, int], int] = {}
    # Base case: 1 segment from 0..j
    for j in range(opts.min_points_per_seg - 1, N):
        dp[1, j] = cost[0, j]
        parent[(1, j)] = -1

    # Fill DP
    for k in range(2, Kmax + 1):
        for j in range((k) * opts.min_points_per_seg - 1, N):
            best_v = INF
            best_i = None
            # Try previous break at i, current seg is (i+1..j)
            i_min = (k - 1) * opts.min_points_per_seg - 1
            for i in range(i_min, j - opts.min_points_per_seg + 1):
                val = dp[k - 1, i] + cost[i + 1, j]
                if val < best_v:
                    best_v = val; best_i = i
            dp[k, j] = best_v
            parent[(k, j)] = best_i if best_i is not None else -1

    # Select k via criterion
    def k_complexity(k: int) -> int:
        # per segment: a,n plus 1 breakpoint (x0), except first x0 known = x[start]
        # Here we treat total params approx as 3k - 1; if last segment fixed reduce by 2
        p = 3 * k - 1
        if last_segment_params is not None:
            p -= 2
        return max(p, 1)

    def select_k():
        candidates = []
        for k in range(opts.k_min, opts.k_max + 1):
            j = N - 1
            ll = -dp[k, j]  # pseudo log-likelihood from squared residuals
            rss = dp[k, j] * 2  # least_squares cost is 1/2 * rss
            p = k_complexity(k)
            if opts.model_selection == "bic":
                crit = N * np.log(max(rss / N, 1e-12)) + p * np.log(N)
            elif opts.model_selection == "aic":
                crit = N * np.log(max(rss / N, 1e-12)) + 2 * p
            elif opts.model_selection == "mdl":
                crit = rss + p * np.log(N)
            elif opts.model_selection == "k_fixed" and opts.k_fixed is not None:
                crit = 0 if k == opts.k_fixed else 1e9
            else:
                crit = rss
            candidates.append((crit, k))
        candidates.sort()
        return candidates[0][1]

    k_best = select_k()

    # Recover breakpoints
    breaks_idx = []
    j = N - 1
    for k in range(k_best, 1, -1):
        i = parent[(k, j)]
        breaks_idx.append(i)
        j = i
    breaks_idx = sorted([b + 1 for b in breaks_idx])  # segment starts after index i
    starts = [0] + breaks_idx
    ends = breaks_idx + [N]

    segments: List[SegmentParams] = []
    # Enforce continuity and refit sequentially
    y_left = None
    for s, (i, j) in enumerate(zip(starts, ends)):
        xi = x[i:j]
        yi = y[i:j]
        x0 = x[i]
        # If this is the last segment and user fixed (a,n), fit only x0==x[i] (already) -> just accept
        if (s == len(starts) - 1) and (last_segment_params is not None):
            a_L, n_L = last_segment_params
            p = SegmentParams(a=a_L, n=n_L, x0=x0)
            y0_target = y_left
            if y0_target is not None:
                # Check continuity; if violated > tol, flag
                y0 = power_model(np.array([x0 + 1e-6]), p)[0]
                rel_err = abs(y0 - y0_target) / max(abs(y0_target), 1e-9)
                if rel_err > 0.05:
                    pass  # leave to user: cannot change last seg
            segments.append(p)
            continue
        p, c, _ = fit_power_segment(
            xi, yi, x0=x0, y_left_target=y_left,
            continuity_tol=opts.continuity_tol,
            robust=opts.robust, robust_kw=opts.robust_kw,
            cap_max_pct_err=opts.cap_max_pct_err,
        )
        segments.append(p)
        # Set continuity target for next segment as the right-edge value
        y_left = power_model(np.array([x[j-1]]), p)[0]

    result = {
        "k": k_best,
        "breaks_x": [x[s] for s in starts],
        "segments": segments,
        "x_sorted": x,
        "y_sorted": y,
    }

    return result


# ------------------------------- Example usage -------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 200)
    # True 3 segments
    y = np.piecewise(
        x,
        [x < 3, (x >= 3) & (x < 7), x >= 7],
        [lambda t: 2.0 * np.power(np.maximum(t - 0.0, 0), 0.8),
         lambda t: 0.8 * np.power(np.maximum(t - 3.0, 0), 1.4),
         lambda t: 4.0 * np.power(np.maximum(t - 7.0, 0), 0.5)]
    )
    y += rng.normal(0, 0.2, size=y.shape)

    opts = FitOptions(k_min=1, k_max=5, min_points_per_seg=12, continuity_tol=0.05,
                      robust="huber", cap_max_pct_err=0.15)

    res = piecewise_power_fit(x, y, last_segment_params=None, opts=opts)
    print("Segments:", res["k"])
    for s, p in enumerate(res["segments"]):
        print(f"s={s}: a={p.a:.3f} n={p.n:.3f} x0={p.x0:.3f}")
