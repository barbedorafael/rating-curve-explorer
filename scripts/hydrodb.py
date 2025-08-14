# viz.py
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

class HydroDB:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)

    # ---------- internals ----------
    def _read_sql(self, sql: str, params: Tuple = (), parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            if parse_dates:
                return pd.read_sql_query(sql, conn, params=params, parse_dates=parse_dates)
            return pd.read_sql_query(sql, conn, params=params)
        
    # ---------- public API ----------
    def plot_timeseries(
        self,
        station: int,
        variable: str,                       # 'level' or 'discharge'
        start_date: str | None = None,       # 'YYYY-MM-DD'
        end_date: str | None = None,         # exclusive upper bound if provided
        rolling: int | None = None,          # optional smoothing window (days)
    ) -> None:
        """
        Plot a daily time series from timeseries_cota ('level') or timeseries_vazao ('discharge').
        """
        variable = variable.lower().strip()
        if variable not in ("level", "discharge"):
            raise ValueError("variable must be 'level' or 'discharge'")

        table = "timeseries_cota" if variable == "level" else "timeseries_vazao"

        sql = f"SELECT date, value FROM {table} WHERE station_id = ?"
        params: list = [station]
        if start_date:
            sql += " AND date >= ?"; params.append(start_date)
        if end_date:
            sql += " AND date < ?";  params.append(end_date)
        sql += " ORDER BY date"

        df = self._read_sql(sql, tuple(params), parse_dates=["date"])
        if df.empty:
            print(f"No data for station {station} in {table}.")
            return

        plt.figure()
        plt.plot(df["date"], df["value"], label="value")
        if rolling and rolling > 1:
            df["roll"] = df["value"].rolling(rolling, min_periods=max(1, rolling // 2)).mean()
            plt.plot(df["date"], df["roll"], label=f"{rolling}-day mean")
        plt.title(f"{'Level' if variable=='level' else 'Discharge'} — station {station}")
        plt.xlabel("Date"); plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_vertical_profile(self, survey_id: int) -> None:
        """
        Plot cross-section (distance vs elevation) for a given survey_id from vertical_profile.
        """
        sql = """
        SELECT distance, elevation
        FROM vertical_profile
        WHERE survey_id = ?
        ORDER BY distance
        """
        df = self._read_sql(sql, (survey_id,))
        if df.empty:
            print(f"No vertical_profile points for survey {survey_id}.")
            return

        x = df["distance"].astype(float).values
        y = df["elevation"].astype(float).values

        plt.figure()
        plt.plot(x, y)
        plt.title(f"Vertical profile — survey {survey_id}")
        plt.xlabel("Distance (m)")
        plt.ylabel("Elevation (cm)")
        plt.tight_layout()
        plt.show()

    # ---------- fitting helpers ----------
    def _fit_powerlaw(self, H, Q, robust=True):
        """
        Nonlinear fit of Q = a * (H - h0)^b.
        Uses SciPy least_squares if available (robust Huber loss), else NumPy fallback.
        Returns (a, b, h0, rmse) or (None, None, None, None).
        """
        H = np.asarray(H, float); Q = np.asarray(Q, float)
        m = np.isfinite(H) & np.isfinite(Q) & (Q > 0)
        H, Q = H[m], Q[m]
        if H.size < 3:
            return None, None, None, None

        try:
            from scipy.optimize import least_squares  # type: ignore
            hmin = float(np.min(H))
            # quick seeds from log-linear with a few h0 guesses
            seeds = np.linspace(hmin - 0.5, np.quantile(H, 0.3), 5)
            a0, b0, h0s = 1.0, 1.0, hmin - 0.1
            for h0_try in seeds:
                x = H - h0_try
                if np.any(x <= 0): 
                    continue
                X = np.log(x); Y = np.log(Q)
                A = np.vstack([np.ones_like(X), X]).T
                c0, c1 = np.linalg.lstsq(A, Y, rcond=None)[0]
                a0, b0, h0s = float(np.exp(c0)), float(c1), float(h0_try)
                break

            def residuals(p):
                a, b, h0 = p
                x = H - h0
                bad = x <= 0
                x[bad] = np.nan
                yhat = a * np.power(x, b)
                r = Q - yhat
                r[~np.isfinite(r)] = 1e6
                r[bad] = 1e6
                return r

            bounds = ([1e-12, 0.1, -np.inf], [np.inf, 6.0, float(np.min(H)) - 1e-6])
            loss = "huber" if robust else "linear"
            res = least_squares(residuals, x0=[a0, b0, h0s], bounds=bounds, loss=loss, f_scale=1.0, max_nfev=2000)
            if not res.success:
                return None, None, None, None
            a, b, h0 = map(float, res.x)
            rmse = float(np.sqrt(np.mean(res.fun**2)))
            return a, b, h0, rmse
        except Exception:
            return self._fit_powerlaw_numpy(H, Q)

    def _draw_rating_segment(self, H_seg, Q_seg, lo, hi, label_prefix=None, robust=True, min_points=3):
        """
        Fit on (H_seg, Q_seg) and draw curve for H in [lo, hi].
        Returns (a,b,h0,rmse) or None if skipped/failed.
        """
        H_seg = np.asarray(H_seg, float); Q_seg = np.asarray(Q_seg, float)
        mask = np.isfinite(H_seg) & np.isfinite(Q_seg) & (Q_seg > 0)
        H_seg, Q_seg = H_seg[mask], Q_seg[mask]
        if H_seg.size < min_points:
            print(f"  (segment {label_prefix or f'{lo:.2f}–{hi:.2f}'}: only {H_seg.size} points)")
            return None

        a, b, h0, rmse = self._fit_powerlaw(H_seg, Q_seg, robust=robust)
        if a is None:
            print(f"  (fit failed {label_prefix or f'{lo:.2f}–{hi:.2f}'})")
            return None

        xs = np.linspace(lo, hi, 200)
        dx = xs - h0
        dx[dx <= 0] = np.nan
        ys = a * np.power(dx, b)
        if np.all(~np.isfinite(ys)):
            print(f"  (degenerate curve {label_prefix or f'{lo:.2f}–{hi:.2f}'})")
            return None

        label = f"{label_prefix or f'{lo:.2f}–{hi:.2f}'}: Q={a:.3g}(H−{h0:.3g})^{b:.3g}  RMSE={rmse:.2g}"
        plt.plot(xs, ys, linewidth=2, label=label)
        return a, b, h0, rmse

    # ---------- plot ----------
    def plot_rating_curve(
        self,
        station: int,
        start_date: str | None = None,
        end_date: str | None = None,
        level_breaks: Iterable[float] | None = None,   # e.g., [X, Y, Z]
        fit: bool = True,
    ) -> None:
        """
        Scatter of discharge vs level from stage_discharge with optional segmented fits.

        level_breaks defines boundaries on H (level). Fits are done on:
            [min(H), b1], (b1, b2], ..., (b_{n-1}, b_n], and by default also (b_n, max(H)]
        Plot orientation: x = H (level), y = Q (discharge).
        """
        sql = """
        SELECT date, time, level AS H, discharge AS Q
        FROM stage_discharge
        WHERE station_id = ?
        """
        params: list = [station]
        if start_date:
            sql += " AND date >= ?"; params.append(start_date)
        if end_date:
            sql += " AND date < ?";  params.append(end_date)
        sql += " AND level IS NOT NULL AND discharge IS NOT NULL ORDER BY date, time"

        df = self._read_sql(sql, tuple(params))
        if df.empty:
            print(f"No stage_discharge data for station {station} in the given range.")
            return

        H = df["H"].astype(float).values
        Q = df["Q"].astype(float).values

        plt.figure()
        plt.scatter(H, Q, alpha=0.85, label="observations")
        plt.title(f"Stage–Discharge — station {station}")
        plt.xlabel("Water level")          # set unit label as needed
        plt.ylabel("Discharge (m³/s)")

        if fit:
            if level_breaks:
                brks = sorted(set(float(x) for x in level_breaks))
                # include final segment up to max(H) so you always see the “last” curve
                edges = [float(np.nanmin(H))] + brks + [float(np.nanmax(H))]
                for i in range(len(edges) - 1):
                    lo, hi = edges[i], edges[i + 1]
                    seg_mask = (H >= lo) & (H <= hi)  # closed segments are fine
                    _ = self._draw_rating_segment(H[seg_mask], Q[seg_mask], lo, hi,
                                                label_prefix=f"{lo:.2f}–{hi:.2f}", robust=True)
            else:
                lo, hi = float(np.nanmin(H)), float(np.nanmax(H))
                _ = self._draw_rating_segment(H, Q, lo, hi, robust=True)

        plt.legend()
        plt.tight_layout()
        plt.show()