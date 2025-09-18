#!/usr/bin/env python3
"""
Inheritance-based Streamlit Dashboard for Rating Curve Data Analysis

Uses object-oriented inheritance to share data between different analysis types:
- BaseStationAnalyzer: Shared data loading and caching
- TimeseriesAnalyzer: Timeseries plots with rating curve indicators
- ScatterAnalyzer: Scatter plots of stage_discharge variables by rating curve
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Rating Curve Explorer",
    page_icon="🌊",
    layout="wide"
)

class BaseStationAnalyzer:
    """
    Base class for station data analysis with shared data loading and caching.
    All analyzers inherit from this to share data efficiently.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)

        # Station and date state
        self._station_id = None
        self._start_date = None
        self._end_date = None

        # Shared cached data
        self._stations_cache = None
        self._measured_data = None  # stage_discharge table
        self._rating_curves = None
        self._timeseries_data = {}  # Cache for different timeseries types

        # Color mapping for rating curves (consistent across all plots)
        self._curve_colors = {}

    def _read_sql(self, sql: str, params: Tuple = (), parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """Execute SQL query with connection management."""
        with sqlite3.connect(self.db_path) as conn:
            if parse_dates:
                return pd.read_sql_query(sql, conn, params=params, parse_dates=parse_dates)
            return pd.read_sql_query(sql, conn, params=params)

    @property
    def stations(self) -> pd.DataFrame:
        """Get list of available stations (cached)."""
        if self._stations_cache is None:
            sql = """
            SELECT DISTINCT s.station_id, s.name, s.river_id
            FROM stations s
            WHERE s.station_id IN (
                SELECT DISTINCT station_id FROM timeseries_cota
                UNION
                SELECT DISTINCT station_id FROM rating_curve
            )
            ORDER BY s.station_id
            """
            self._stations_cache = self._read_sql(sql)
        return self._stations_cache

    def set_station(self, station_id: int, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """Set current station and date range, clearing cached data if changed."""
        if (self._station_id != station_id or
            self._start_date != start_date or
            self._end_date != end_date):

            self._station_id = station_id
            self._start_date = start_date
            self._end_date = end_date

            # Clear cached data when parameters change
            self._measured_data = None
            self._rating_curves = None
            self._timeseries_data = {}

    @property
    def measured_data(self) -> pd.DataFrame:
        """Get measured discharge points with ALL variables (stage_discharge table)."""
        if self._measured_data is None:
            self._measured_data = self._load_measured_data()
        return self._measured_data

    @property
    def rating_curves(self) -> pd.DataFrame:
        """Get rating curve parameters for current station."""
        if self._rating_curves is None:
            self._rating_curves = self._load_rating_curves()
            self._setup_curve_colors()
        return self._rating_curves

    def _load_measured_data(self) -> pd.DataFrame:
        """Load ALL measured data from stage_discharge table with date filtering."""
        if self._station_id is None:
            return pd.DataFrame()

        sql = """
        SELECT date, time, level, discharge, area, velocity, width, depth, consistency
        FROM stage_discharge
        WHERE station_id = ?
        AND level IS NOT NULL AND discharge IS NOT NULL
        """
        params = [self._station_id]

        if self._start_date:
            sql += " AND date >= ?"
            params.append(self._start_date.strftime('%Y-%m-%d'))
        if self._end_date:
            sql += " AND date <= ?"
            params.append(self._end_date.strftime('%Y-%m-%d'))

        sql += " ORDER BY date, time"

        df = self._read_sql(sql, tuple(params), parse_dates=['date'])

        # Calculate derived variables
        if not df.empty:
            # Calculate area × velocity product for one of the scatter plots
            df['area_velocity'] = df['area'] * df['velocity']

        return df

    def _load_rating_curves(self) -> pd.DataFrame:
        """Load rating curve parameters for current station."""
        if self._station_id is None:
            return pd.DataFrame()

        sql = """
        SELECT segment_number, start_date, end_date, h_min, h_max,
               h0_param, a_param, n_param
        FROM rating_curve
        WHERE station_id = ?
        ORDER BY start_date DESC, segment_number ASC
        """

        df = self._read_sql(sql, (self._station_id,), parse_dates=['start_date', 'end_date'])
        if not df.empty:
            # Handle NULL end_date
            df['end_date'] = df['end_date'].fillna(pd.Timestamp('2099-12-31'))

            # Create formatted curve ID: "start_date: segment_number"
            df['curve_id'] = df['start_date'].dt.strftime('%Y-%m-%d') + ': ' + df['segment_number']

        return df

    def _setup_curve_colors(self):
        """Setup consistent color mapping for rating curves."""
        if not self.rating_curves.empty:
            # Use seaborn color palette for consistent colors
            n_curves = len(self.rating_curves)
            colors = sns.color_palette("tab10", n_curves)

            # Clear existing colors for new station
            self._curve_colors = {}

            for i, (_, curve) in enumerate(self.rating_curves.iterrows()):
                self._curve_colors[curve['curve_id']] = colors[i]

    def get_data_by_rating_curve(self, selected_curves: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Filter measured data by rating curve date and level constraints.
        Returns dict mapping curve_id to filtered dataframe.
        """
        if self.measured_data.empty or self.rating_curves.empty:
            return {}

        curve_data = {}

        for _, curve in self.rating_curves.iterrows():
            curve_id = curve['curve_id']  # Use new formatted curve_id

            # Skip if not in selected curves
            if selected_curves and curve_id not in selected_curves:
                continue

            # Filter by date range
            date_mask = (
                (self.measured_data['date'] >= curve['start_date']) &
                (self.measured_data['date'] <= curve['end_date'])
            )

            # Filter by level range
            level_mask = (
                (self.measured_data['level'] >= curve['h_min']) &
                (self.measured_data['level'] <= curve['h_max'])
            )

            # Combined filter
            combined_mask = date_mask & level_mask
            filtered_data = self.measured_data[combined_mask].copy()

            if not filtered_data.empty:
                curve_data[curve_id] = filtered_data

        return curve_data

    def get_timeseries_data(self, variable: str) -> pd.DataFrame:
        """Get timeseries data (level or discharge) with caching."""
        if variable not in self._timeseries_data:
            self._timeseries_data[variable] = self._load_timeseries_data(variable)
        return self._timeseries_data[variable]

    def _load_timeseries_data(self, variable: str) -> pd.DataFrame:
        """Load timeseries data (level or discharge) with date filtering."""
        if self._station_id is None:
            return pd.DataFrame()

        table = "timeseries_cota" if variable == "level" else "timeseries_vazao"
        value_col = "level" if variable == "level" else "discharge"

        sql = f"SELECT date, value as {value_col}, status FROM {table} WHERE station_id = ?"
        params = [self._station_id]

        if self._start_date:
            sql += " AND date >= ?"
            params.append(self._start_date.strftime('%Y-%m-%d'))
        if self._end_date:
            sql += " AND date <= ?"
            params.append(self._end_date.strftime('%Y-%m-%d'))

        sql += " ORDER BY date"

        df = self._read_sql(sql, tuple(params), parse_dates=['date'])
        return df

    def get_data_summary(self) -> dict:
        """Get summary statistics for current station data."""
        return {
            'measured_points': len(self.measured_data),
            'rating_curves': len(self.rating_curves),
            'level_records': len(self.get_timeseries_data('level')),
            'discharge_records': len(self.get_timeseries_data('discharge'))
        }


class TimeseriesAnalyzer(BaseStationAnalyzer):
    """Analyzer for timeseries plots with rating curve indicators."""

    def _fill_missing_dates(self, data: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Fill missing dates with NaN to prevent interpolation."""
        if data.empty:
            return data

        # Create complete date range
        date_range = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')

        # Create DataFrame with all dates
        complete_df = pd.DataFrame({'date': date_range})

        # Merge with original data
        filled_data = complete_df.merge(data, on='date', how='left')
        return filled_data

    def _get_plot_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get plot date range based on filters or data."""
        if self._start_date and self._end_date:
            return pd.Timestamp(self._start_date), pd.Timestamp(self._end_date)

        # Find earliest and latest dates from all data sources
        dates = []
        for variable in ['level', 'discharge']:
            data = self.get_timeseries_data(variable)
            if not data.empty:
                dates.extend([data['date'].min(), data['date'].max()])

        if not self.measured_data.empty:
            dates.extend([self.measured_data['date'].min(), self.measured_data['date'].max()])

        if dates:
            return min(dates), max(dates)
        else:
            return pd.Timestamp('1960-01-01'), pd.Timestamp('2026-01-01')

    def _format_x_axis(self, ax, plot_start: pd.Timestamp, plot_end: pd.Timestamp):
        """Format x-axis with appropriate tick spacing."""
        date_range_days = (plot_end - plot_start).days

        if date_range_days <= 1825:  # Less than 5 years
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif date_range_days <= 5475:  # Less than 15 years
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:  # More than 15 years
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    def plot_timeseries(self, plot_type: str, figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """Create timeseries plot with rating curve indicators."""
        if self._station_id is None:
            raise ValueError("No station selected. Use set_station() first.")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get appropriate data
        data = self.get_timeseries_data(plot_type.lower())

        if plot_type == "Level":
            value_col = 'level'
            ylabel = 'Water Level (cm)'
            color = 'b'
            label = 'Water Level'
        else:
            value_col = 'discharge'
            ylabel = 'Discharge (m³/s)'
            color = 'g'
            label = 'Discharge'

        # Get plot date range
        plot_start, plot_end = self._get_plot_date_range()

        # Fill missing dates to prevent interpolation
        if not data.empty:
            filled_data = self._fill_missing_dates(data, value_col)

            # Plot main timeseries
            ax.plot(filled_data['date'], filled_data[value_col],
                   f'{color}-', linewidth=1, alpha=0.7, label=label)

        # Set axis properties
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis='y', labelcolor=color)

        # Add rating curve validity indicators
        for _, curve in self.rating_curves.iterrows():
            # Only show indicators within the plot date range
            if curve['end_date'] >= plot_start and curve['start_date'] <= plot_end:
                # Vertical lines for date ranges
                if curve['start_date'] >= plot_start:
                    ax.axvline(curve['start_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
                if curve['end_date'] < pd.Timestamp('2099-01-01') and curve['end_date'] <= plot_end:
                    ax.axvline(curve['end_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)

                # Horizontal lines for level limits (limited to date range)
                line_start = max(curve['start_date'], plot_start)
                line_end = min(curve['end_date'], plot_end)

                ax.hlines(curve['h_min'], line_start, line_end,
                         colors='orange', linestyles=':', alpha=0.5, linewidth=1)
                ax.hlines(curve['h_max'], line_start, line_end,
                         colors='orange', linestyles=':', alpha=0.5, linewidth=1)

                # Add text annotation for segment below the line
                mid_date = line_start + (line_end - line_start) / 2
                y_pos = curve['h_min']
                ax.text(mid_date, y_pos, f"Seg {curve['segment_number']}",
                       rotation=0, fontsize=8, ha='center', va='bottom', color='red')

        # Add measured points
        if not self.measured_data.empty:
            good_points = self.measured_data[self.measured_data['consistency'] == 1]
            poor_points = self.measured_data[self.measured_data['consistency'] != 1]

            if plot_type == "Level":
                y_values_good = good_points['level']
                y_values_poor = poor_points['level']
            else:
                y_values_good = good_points['discharge']
                y_values_poor = poor_points['discharge']

            if not good_points.empty:
                ax.scatter(good_points['date'], y_values_good,
                          c='darkgreen', s=30, alpha=0.8, label='Measurement Expeditions', zorder=5)

            if not poor_points.empty:
                ax.scatter(poor_points['date'], y_values_poor,
                          c='orange', s=30, alpha=0.8, label='Measurement Expeditions (Poor)', zorder=5)

        # Set axis limits and formatting
        ax.set_xlim(plot_start, plot_end)
        ax.set_xlabel('Date')

        # Format x-axis with dynamic spacing
        self._format_x_axis(ax, plot_start, plot_end)
        plt.xticks(rotation=45)

        # Add legend and grid
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Title
        title_text = f'Station {self._station_id} - {plot_type} Timeseries\n'
        title_text += 'Red dashed lines: Rating curve periods | Orange dotted lines: Level limits'

        plt.title(title_text)
        plt.tight_layout()

        return fig


class ScatterAnalyzer(BaseStationAnalyzer):
    """Analyzer for scatter plots of stage_discharge variables by rating curve."""

    # Configuration for different scatter plot types
    PLOT_CONFIGS = {
        'discharge_vs_level': {
            'x': 'discharge', 'y': 'level',
            'xlabel': 'Discharge (m³/s)', 'ylabel': 'Level (cm)',
            'title': 'Discharge vs Level'
        },
        'area_vs_level': {
            'x': 'area', 'y': 'level',
            'xlabel': 'Cross-sectional Area (m²)', 'ylabel': 'Level (cm)',
            'title': 'Cross-sectional Area vs Level'
        },
        'area_velocity_vs_discharge': {
            'x': 'area_velocity', 'y': 'discharge',
            'xlabel': 'Area × Velocity (m³/s)', 'ylabel': 'Discharge (m³/s)',
            'title': 'Area × Velocity vs Discharge'
        },
        'velocity_vs_level': {
            'x': 'velocity', 'y': 'level',
            'xlabel': 'Velocity (m/s)', 'ylabel': 'Level (cm)',
            'title': 'Velocity vs Level'
        }
    }

    def plot_scatter(self, plot_type: str, selected_curves: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create scatter plot of stage_discharge variables colored by rating curve.

        Args:
            plot_type: One of the keys in PLOT_CONFIGS
            selected_curves: List of curve segment_numbers to include
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        if self._station_id is None:
            raise ValueError("No station selected. Use set_station() first.")

        if plot_type not in self.PLOT_CONFIGS:
            raise ValueError(f"Unknown plot type: {plot_type}")

        config = self.PLOT_CONFIGS[plot_type]

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get data filtered by rating curves
        curve_data = self.get_data_by_rating_curve(selected_curves)

        if not curve_data:
            ax.text(0.5, 0.5, 'No data available for selected curves',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlabel(config['xlabel'])
            ax.set_ylabel(config['ylabel'])
            ax.set_title(f"Station {self._station_id} - {config['title']}")
            return fig

        # Plot each rating curve with different color
        for curve_id, data in curve_data.items():
            if data.empty:
                continue

            # Check if required columns exist and have valid data
            x_col, y_col = config['x'], config['y']
            if x_col not in data.columns or y_col not in data.columns:
                continue

            # Filter out NaN values
            mask = data[x_col].notna() & data[y_col].notna()
            if not mask.any():
                continue

            plot_data = data[mask]

            # Get color for this curve
            color = self._curve_colors.get(curve_id, 'gray')

            # Create scatter plot
            ax.scatter(plot_data[x_col], plot_data[y_col],
                      c=[color], alpha=0.7, s=40,
                      label=f'Curve {curve_id}', zorder=5)

        # Formatting
        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel(config['ylabel'])
        ax.set_title(f"Station {self._station_id} - {config['title']}\n" \
                    f"Data colored by rating curve segment")

        # Add legend if there are multiple curves
        if len(curve_data) > 1:
            ax.legend(loc='best')

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig


class ProfileAnalyzer(BaseStationAnalyzer):
    """Analyzer for vertical profile plots with rating curve filtering."""

    def get_vertical_profiles(self) -> pd.DataFrame:
        """Get all vertical profile survey metadata for current station."""
        if self._station_id is None:
            return pd.DataFrame()

        sql = """
        SELECT vps.survey_id, vps.date, vps.num_survey, vps.section_type,
               vps.num_verticals, vps.dist_min, vps.dist_max, vps.elev_min, vps.elev_max
        FROM vertical_profile_survey vps
        WHERE vps.station_id = ?
        ORDER BY vps.date DESC
        """

        df = self._read_sql(sql, (self._station_id,), parse_dates=['date'])
        return df

    def get_profile_data(self, survey_ids: List[int]) -> Dict[int, pd.DataFrame]:
        """Get vertical profile measurement data for selected surveys."""
        if not survey_ids:
            return {}

        # Create placeholders for the IN clause
        placeholders = ','.join(['?'] * len(survey_ids))
        sql = f"""
        SELECT survey_id, distance, elevation
        FROM vertical_profile
        WHERE survey_id IN ({placeholders})
        ORDER BY survey_id, distance
        """

        df = self._read_sql(sql, tuple(survey_ids))

        # Group by survey_id
        profile_data = {}
        for survey_id in survey_ids:
            survey_data = df[df['survey_id'] == survey_id].copy()
            if not survey_data.empty:
                profile_data[survey_id] = survey_data

        return profile_data

    def get_primary_rating_curves(self) -> pd.DataFrame:
        """Get primary rating curves (segment numbers starting with '01/')."""
        if self.rating_curves.empty:
            return pd.DataFrame()

        # Filter to only segment numbers starting with "01/"
        primary_curves = self.rating_curves[
            self.rating_curves['segment_number'].str.startswith('01/')
        ].copy()

        return primary_curves

    def filter_profiles_by_rating_curve(self, selected_curve_id: Optional[str] = None) -> List[int]:
        """Filter vertical profiles by a single rating curve date range."""
        profiles = self.get_vertical_profiles()

        if profiles.empty:
            return []

        if not selected_curve_id:
            # Return all survey IDs if no curve selected
            return profiles['survey_id'].tolist()

        # Find the selected curve
        selected_curve = self.rating_curves[
            self.rating_curves['curve_id'] == selected_curve_id
        ]

        if selected_curve.empty:
            return []

        curve = selected_curve.iloc[0]

        # Filter profiles by date range
        filtered_survey_ids = []
        for _, profile in profiles.iterrows():
            profile_date = profile['date']
            if curve['start_date'] <= profile_date <= curve['end_date']:
                filtered_survey_ids.append(profile['survey_id'])

        return filtered_survey_ids

    def plot_vertical_profiles(self, selected_curve_id: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create vertical profile plots colored by survey_id with rating curve filtering.

        Args:
            selected_curve_id: Single curve_id to filter profiles by date
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        if self._station_id is None:
            raise ValueError("No station selected. Use set_station() first.")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get filtered survey IDs based on rating curve selection
        filtered_survey_ids = self.filter_profiles_by_rating_curve(selected_curve_id)

        if not filtered_survey_ids:
            ax.text(0.5, 0.5, 'No vertical profiles available for selected rating curve period',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Elevation (cm)')
            ax.set_title(f"Station {self._station_id} - Vertical Profiles")
            return fig

        # Get profile data
        profile_data = self.get_profile_data(filtered_survey_ids)
        profiles_meta = self.get_vertical_profiles()

        if not profile_data:
            ax.text(0.5, 0.5, 'No profile measurement data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Elevation (cm)')
            ax.set_title(f"Station {self._station_id} - Vertical Profiles")
            return fig

        # Create color palette for surveys
        n_surveys = len(profile_data)
        colors = sns.color_palette("tab10", n_surveys)

        # Plot each survey with different color
        for i, (survey_id, data) in enumerate(profile_data.items()):
            # Get survey metadata for labeling
            survey_meta = profiles_meta[profiles_meta['survey_id'] == survey_id]
            if not survey_meta.empty:
                survey_date = survey_meta.iloc[0]['date'].strftime('%Y-%m-%d')
                label = f"Survey {survey_id} ({survey_date})"
            else:
                label = f"Survey {survey_id}"

            ax.plot(data['distance'], data['elevation'],
                   color=colors[i], linewidth=2, alpha=0.8, label=label)

        # Add h_max horizontal lines from ALL rating curve segments (not just the selected one)
        if selected_curve_id and not self.rating_curves.empty:
            # Get the selected primary curve to determine the date range
            selected_curve = self.rating_curves[
                self.rating_curves['curve_id'] == selected_curve_id
            ]

            if not selected_curve.empty:
                primary_curve = selected_curve.iloc[0]

                # Show h_max for all segments that overlap with the selected curve's date range
                for _, curve in self.rating_curves.iterrows():
                    # Check if this curve overlaps with the selected curve's date range
                    overlaps = (
                        curve['start_date'] <= primary_curve['end_date'] and
                        curve['end_date'] >= primary_curve['start_date']
                    )

                    if overlaps:
                        # Use the curve color for consistency
                        curve_color = self._curve_colors.get(curve['curve_id'], 'gray')

                        # Draw h_max line
                        ax.axhline(curve['h_max'], color=curve_color, linestyle='--',
                                 alpha=0.7, linewidth=2)

                        # Add text label on the right side of the plot
                        ax.text(0.98, curve['h_max'], f"{curve['segment_number']}: {curve['h_max']:.0f}cm",
                               transform=ax.get_yaxis_transform(),
                               verticalalignment='bottom', horizontalalignment='right',
                               fontsize=9, color=curve_color, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=curve_color))

        # Add stage_discharge campaign dots (level values as dots in the middle of plot)
        # Only show dots when a specific rating curve is selected (not for "Show All")
        if selected_curve_id and not self.measured_data.empty:
            # Get the selected curve to filter by date range
            selected_curve = self.rating_curves[
                self.rating_curves['curve_id'] == selected_curve_id
            ]

            if not selected_curve.empty:
                curve = selected_curve.iloc[0]

                # Filter measured data by the selected curve's date range
                filtered_measurements = self.measured_data[
                    (self.measured_data['date'] >= curve['start_date']) &
                    (self.measured_data['date'] <= curve['end_date'])
                ]

                if not filtered_measurements.empty:
                    # Get x-axis range to position dots in the middle
                    xlim = ax.get_xlim()
                    x_middle = (xlim[0] + xlim[1]) / 2

                    # Plot level values from stage_discharge as dots
                    for _, measurement in filtered_measurements.iterrows():
                        ax.scatter(x_middle, measurement['level'],
                                  c='red', s=20, alpha=0.6, zorder=10)

        # Formatting
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Elevation (cm)')

        title_text = f"Station {self._station_id} - Vertical Cross-Section Profiles"
        if selected_curve_id:
            # Get curve info for title
            selected_curve = self.rating_curves[
                self.rating_curves['curve_id'] == selected_curve_id
            ]
            if not selected_curve.empty:
                curve = selected_curve.iloc[0]
                start_str = curve['start_date'].strftime('%Y-%m-%d')
                end_str = curve['end_date'].strftime('%Y-%m-%d') if curve['end_date'] < pd.Timestamp('2099-01-01') else 'Current'
                title_text += f"\nFiltered by: {start_str} to {end_str}"

        ax.set_title(title_text)

        # Add legend
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_selected_vertical_profiles(self, selected_curve_id: Optional[str] = None,
                                      selected_survey_ids: List[int] = None,
                                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create vertical profile plots for specifically selected surveys.

        Args:
            selected_curve_id: Single curve_id for h_max lines and campaign dots
            selected_survey_ids: List of survey IDs to plot
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        if self._station_id is None:
            raise ValueError("No station selected. Use set_station() first.")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if not selected_survey_ids:
            ax.text(0.5, 0.5, 'No surveys selected for display',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Elevation (cm)')
            ax.set_title(f"Station {self._station_id} - Vertical Profiles")
            return fig

        # Get profile data for selected surveys only
        profile_data = self.get_profile_data(selected_survey_ids)
        profiles_meta = self.get_vertical_profiles()

        if not profile_data:
            ax.text(0.5, 0.5, 'No profile measurement data available for selected surveys',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Elevation (cm)')
            ax.set_title(f"Station {self._station_id} - Vertical Profiles")
            return fig

        # Create color palette for surveys
        n_surveys = len(profile_data)
        colors = sns.color_palette("tab10", n_surveys)

        # Plot each selected survey with different color
        for i, (survey_id, data) in enumerate(profile_data.items()):
            # Get survey metadata for labeling
            survey_meta = profiles_meta[profiles_meta['survey_id'] == survey_id]
            if not survey_meta.empty:
                survey_date = survey_meta.iloc[0]['date'].strftime('%Y-%m-%d')
                label = f"Survey {survey_id} ({survey_date})"
            else:
                label = f"Survey {survey_id}"

            ax.plot(data['distance'], data['elevation'],
                   color=colors[i], linewidth=2, alpha=0.8, label=label)

        # Add h_max horizontal lines from ALL rating curve segments (not just the selected one)
        if selected_curve_id and not self.rating_curves.empty:
            # Get the selected primary curve to determine the date range
            selected_curve = self.rating_curves[
                self.rating_curves['curve_id'] == selected_curve_id
            ]

            if not selected_curve.empty:
                primary_curve = selected_curve.iloc[0]

                # Show h_max for all segments that overlap with the selected curve's date range
                for _, curve in self.rating_curves.iterrows():
                    # Check if this curve overlaps with the selected curve's date range
                    overlaps = (
                        curve['start_date'] <= primary_curve['end_date'] and
                        curve['end_date'] >= primary_curve['start_date']
                    )

                    if overlaps:
                        # Use the curve color for consistency
                        curve_color = self._curve_colors.get(curve['curve_id'], 'gray')

                        # Draw h_max line
                        ax.axhline(curve['h_max'], color=curve_color, linestyle='--',
                                 alpha=0.7, linewidth=2)

                        # Add text label on the right side of the plot
                        ax.text(0.98, curve['h_max'], f"{curve['segment_number']}: {curve['h_max']:.0f}cm",
                               transform=ax.get_yaxis_transform(),
                               verticalalignment='bottom', horizontalalignment='right',
                               fontsize=9, color=curve_color, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=curve_color))

        # Add stage_discharge campaign dots (level values as dots in the middle of plot)
        # Only show dots when a specific rating curve is selected (not for "Show All")
        if selected_curve_id and not self.measured_data.empty:
            # Get the selected curve to filter by date range
            selected_curve = self.rating_curves[
                self.rating_curves['curve_id'] == selected_curve_id
            ]

            if not selected_curve.empty:
                curve = selected_curve.iloc[0]

                # Filter measured data by the selected curve's date range
                filtered_measurements = self.measured_data[
                    (self.measured_data['date'] >= curve['start_date']) &
                    (self.measured_data['date'] <= curve['end_date'])
                ]

                if not filtered_measurements.empty:
                    # Get x-axis range to position dots in the middle
                    xlim = ax.get_xlim()
                    x_middle = (xlim[0] + xlim[1]) / 2

                    # Plot level values from stage_discharge as dots
                    for _, measurement in filtered_measurements.iterrows():
                        ax.scatter(x_middle, measurement['level'],
                                  c='red', s=20, alpha=0.6, zorder=10)

        # Formatting
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Elevation (cm)')

        title_text = f"Station {self._station_id} - Vertical Cross-Section Profiles"
        if selected_curve_id:
            # Get curve info for title
            selected_curve = self.rating_curves[
                self.rating_curves['curve_id'] == selected_curve_id
            ]
            if not selected_curve.empty:
                curve = selected_curve.iloc[0]
                start_str = curve['start_date'].strftime('%Y-%m-%d')
                end_str = curve['end_date'].strftime('%Y-%m-%d') if curve['end_date'] < pd.Timestamp('2099-01-01') else 'Current'
                title_text += f"\nFiltered by: {start_str} to {end_str}"

        ax.set_title(title_text)

        # Add legend
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class DashboardApp:
    """Main Streamlit dashboard application using composition."""

    def __init__(self, db_path: str = "data/hydrodata.sqlite"):
        # Use composition - all analyzers share the same base data
        self.timeseries_analyzer = TimeseriesAnalyzer(db_path)
        self.scatter_analyzer = ScatterAnalyzer(db_path)
        self.profile_analyzer = ProfileAnalyzer(db_path)

    def _sync_analyzers(self, station_id: int, start_date: Optional[datetime], end_date: Optional[datetime]):
        """Sync station and date settings across all analyzers."""
        self.timeseries_analyzer.set_station(station_id, start_date, end_date)
        self.scatter_analyzer.set_station(station_id, start_date, end_date)
        self.profile_analyzer.set_station(station_id, start_date, end_date)

    def run(self):
        """Run the Streamlit dashboard."""
        st.title("🌊 Rating Curve Data Dashboard")
        st.markdown("Advanced analysis tool with timeseries and scatter plot capabilities")

        # Sidebar controls
        st.sidebar.header("Station & Date Controls")

        try:
            # Station selection (using timeseries analyzer to get stations)
            stations = self.timeseries_analyzer.stations

            if stations.empty:
                st.error("No stations found in database")
                return

            station_options = {f"{row['station_id']} - {row['name']}": row['station_id']
                             for _, row in stations.iterrows()}

            selected_station = st.sidebar.selectbox(
                "Select Station",
                options=list(station_options.keys())
            )

            station_id = station_options[selected_station]

            # Date range selection
            st.sidebar.subheader("Date Range")

            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=None,
                                         min_value=datetime(1900, 1, 1),
                                         max_value=datetime(2099, 12, 31))
            with col2:
                end_date = st.date_input("End Date", value=None,
                                       min_value=datetime(1900, 1, 1),
                                       max_value=datetime(2099, 12, 31))

            # Sync both analyzers
            with st.spinner("Loading station data..."):
                self._sync_analyzers(station_id, start_date, end_date)
                summary = self.timeseries_analyzer.get_data_summary()

            # Display data summary
            st.subheader("Data Summary")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Level Records", summary['level_records'])
            with col2:
                st.metric("Discharge Records", summary['discharge_records'])
            with col3:
                st.metric("Measured Points", summary['measured_points'])
            with col4:
                st.metric("Rating Curves", summary['rating_curves'])

            # Main content in tabs
            tab1, tab2, tab3 = st.tabs(["📈 Timeseries Analysis", "📊 Scatter Analysis", "🏔️ Vertical Profiles"])

            with tab1:
                self._render_timeseries_tab()

            with tab2:
                self._render_scatter_tab()

            with tab3:
                self._render_profile_tab()

        except Exception as e:
            st.error(f"Error loading data: {e}")

    def _render_timeseries_tab(self):
        """Render the timeseries analysis tab."""
        st.subheader("Timeseries with Rating Curve Indicators")

        # Plot type selection
        plot_type = st.radio(
            "Select data to plot:",
            options=["Level", "Discharge"],
            index=0,
            horizontal=True
        )

        # Check if data is available
        data_available = (
            (plot_type == "Level" and len(self.timeseries_analyzer.get_timeseries_data('level')) > 0) or
            (plot_type == "Discharge" and len(self.timeseries_analyzer.get_timeseries_data('discharge')) > 0)
        )

        if data_available:
            fig = self.timeseries_analyzer.plot_timeseries(plot_type)
            st.pyplot(fig)

            # Show rating curve table
            if len(self.timeseries_analyzer.rating_curves) > 0:
                with st.expander("Rating Curve Parameters"):
                    display_curves = self.timeseries_analyzer.rating_curves.copy()
                    display_curves['start_date'] = display_curves['start_date'].dt.strftime('%Y-%m-%d')
                    display_curves['end_date'] = display_curves['end_date'].dt.strftime('%Y-%m-%d')
                    display_curves['end_date'] = display_curves['end_date'].replace('2099-12-31', 'Current')
                    st.dataframe(display_curves, use_container_width=True)
        else:
            st.warning(f"No {plot_type.lower()} data available for this station")

    def _render_scatter_tab(self):
        """Render the scatter analysis tab."""
        st.subheader("Scatter Plots by Rating Curve")

        if len(self.scatter_analyzer.measured_data) == 0:
            st.warning("No measured data (stage_discharge) available for scatter plots")
            return

        if len(self.scatter_analyzer.rating_curves) == 0:
            st.warning("No rating curves available for filtering")
            return

        col1, col2 = st.columns([1, 3])

        with col1:
            # Plot type selection
            plot_options = {
                'discharge_vs_level': 'Discharge vs Level',
                'area_vs_level': 'Area vs Level',
                'area_velocity_vs_discharge': 'Area×Velocity vs Discharge',
                'velocity_vs_level': 'Velocity vs Level'
            }

            selected_plot = st.selectbox(
                "Scatter Plot Type",
                options=list(plot_options.keys()),
                format_func=lambda x: plot_options[x]
            )

            # Rating curve selection with checkboxes
            st.subheader("Rating Curve Selection")
            available_curves = self.scatter_analyzer.rating_curves['curve_id'].tolist()

            # Create checkboxes for each curve
            selected_curves = []

            # Add "Select All" checkbox
            select_all = st.checkbox("Select All", value=True)

            st.write("**Available Curves:**")
            for curve_id in available_curves:
                # Use select_all to determine default state
                default_checked = select_all

                # Create individual checkbox
                is_checked = st.checkbox(
                    curve_id,
                    value=default_checked,
                    key=f"curve_{curve_id}"
                )

                if is_checked:
                    selected_curves.append(curve_id)

        with col2:
            if selected_curves:
                # Create scatter plot
                fig = self.scatter_analyzer.plot_scatter(selected_plot, selected_curves)
                st.pyplot(fig)

                # Data filtering info below the plot
                curve_data = self.scatter_analyzer.get_data_by_rating_curve(selected_curves)
                total_points = sum(len(df) for df in curve_data.values())

                # Show detailed breakdown
                st.subheader("Data Summary")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric("Total Points", total_points)
                    st.metric("Selected Curves", len(curve_data))

                with col_b:
                    # Points per curve breakdown
                    if curve_data:
                        st.write("**Points per Curve:**")
                        for curve_id, data in curve_data.items():
                            st.write(f"• {curve_id}: {len(data)} points")

            else:
                st.info("Please select at least one rating curve to display the scatter plot")

    def _render_profile_tab(self):
        """Render the vertical profiles analysis tab."""
        st.subheader("Vertical Cross-Section Profiles")

        # Check if profile data is available
        profiles = self.profile_analyzer.get_vertical_profiles()

        if profiles.empty:
            st.warning("No vertical profile data available for this station")
            return

        # Get primary rating curves (those starting with "01/")
        primary_curves = self.profile_analyzer.get_primary_rating_curves()

        if primary_curves.empty:
            st.warning("No primary rating curves (segment numbers starting with '01/') found for this station")
            return

        col1, col2 = st.columns([1, 3])

        with col1:
            # Rating curve selection using selectbox
            st.subheader("Filter by Rating Curve")

            # Create options for selectbox
            curve_options = {"Show All": None}
            for _, curve in primary_curves.iterrows():
                start_date_str = curve['start_date'].strftime('%Y-%m-%d')
                end_date_str = curve['end_date'].strftime('%Y-%m-%d') if curve['end_date'] < pd.Timestamp('2099-01-01') else 'Current'
                curve_key = f"Rating Curve of {start_date_str} to {end_date_str}"
                curve_options[curve_key] = curve['curve_id']

            selected_curve_key = st.selectbox(
                "Select rating curve to filter profiles:",
                options=list(curve_options.keys()),
                index=0
            )

            selected_curve_id = curve_options[selected_curve_key]

            # Show profile summary and selection
            st.subheader("Available Profiles")
            if selected_curve_id is None:
                # Show all profiles
                available_survey_ids = profiles['survey_id'].tolist()
            else:
                # Filter by selected curve
                available_survey_ids = self.profile_analyzer.filter_profiles_by_rating_curve(selected_curve_id)

            if available_survey_ids:
                st.metric("Available Surveys", len(available_survey_ids))

                # Survey selection with checkboxes
                st.write("**Select Surveys to Display:**")

                # Add "Select All" checkbox
                select_all_surveys = st.checkbox("Select All Surveys", value=True, key="profile_select_all_surveys")

                selected_survey_ids = []
                available_profiles = profiles[profiles['survey_id'].isin(available_survey_ids)]

                for _, profile in available_profiles.iterrows():
                    survey_id = profile['survey_id']
                    date_str = profile['date'].strftime('%Y-%m-%d')
                    survey_label = f"Survey {survey_id} ({date_str})"

                    # Use select_all to determine default state
                    default_checked = select_all_surveys

                    # Create individual checkbox
                    is_checked = st.checkbox(
                        survey_label,
                        value=default_checked,
                        key=f"survey_{survey_id}"
                    )

                    if is_checked:
                        selected_survey_ids.append(survey_id)

                # Update filtered_survey_ids to only include selected ones
                filtered_survey_ids = selected_survey_ids
            else:
                st.warning("No profiles match selected rating curve period")
                filtered_survey_ids = []

        with col2:
            if filtered_survey_ids:
                # Create vertical profile plot with only selected surveys
                # Override the plot method to use specific survey IDs
                fig = self.profile_analyzer.plot_selected_vertical_profiles(
                    selected_curve_id, filtered_survey_ids
                )
                st.pyplot(fig)

                # Profile data summary below plot
                st.subheader("Profile Data Summary")
                profile_data = self.profile_analyzer.get_profile_data(filtered_survey_ids)

                # Count discharge measurement campaigns
                if selected_curve_id:
                    # Filter campaigns by selected curve's date range
                    selected_curve = self.profile_analyzer.rating_curves[
                        self.profile_analyzer.rating_curves['curve_id'] == selected_curve_id
                    ]
                    if not selected_curve.empty:
                        curve = selected_curve.iloc[0]
                        filtered_campaigns = self.profile_analyzer.measured_data[
                            (self.profile_analyzer.measured_data['date'] >= curve['start_date']) &
                            (self.profile_analyzer.measured_data['date'] <= curve['end_date'])
                        ]
                        campaign_count = len(filtered_campaigns)
                    else:
                        campaign_count = 0
                else:
                    # Show all campaigns when "Show All" is selected
                    campaign_count = len(self.profile_analyzer.measured_data)

                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric("Total Profiles", len(profile_data))
                    st.metric("Discharge Campaigns", campaign_count)

                with col_b:
                    # Points per profile breakdown
                    if profile_data:
                        st.write("**Points per Profile:**")
                        for survey_id, data in profile_data.items():
                            # Get survey date for display
                            survey_meta = profiles[profiles['survey_id'] == survey_id]
                            if not survey_meta.empty:
                                date_str = survey_meta.iloc[0]['date'].strftime('%Y-%m-%d')
                                st.write(f"• Survey {survey_id} ({date_str}): {len(data)} points")
                            else:
                                st.write(f"• Survey {survey_id}: {len(data)} points")

            else:
                st.info("Select rating curves to filter and display vertical profiles")


def main():
    """Entry point for the dashboard."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()