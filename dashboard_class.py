#!/usr/bin/env python3
"""
Class-based Streamlit Dashboard for Rating Curve Data Cleaning

Uses a clean object-oriented architecture similar to HydroDB class.
Provides better state management, caching, and code organization.
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
from functools import lru_cache

# Page configuration
st.set_page_config(
    page_title="Rating Curve Explorer",
    page_icon="🌊",
    layout="wide"
)

class StationAnalyzer:
    """
    A class for analyzing hydrological station data with rating curve indicators.
    Inspired by HydroDB class architecture.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._station_id = None
        self._level_data = None
        self._discharge_data = None
        self._measured_data = None
        self._rating_curves = None
        self._start_date = None
        self._end_date = None

    def _read_sql(self, sql: str, params: Tuple = (), parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """Execute SQL query with connection management."""
        with sqlite3.connect(self.db_path) as conn:
            if parse_dates:
                return pd.read_sql_query(sql, conn, params=params, parse_dates=parse_dates)
            return pd.read_sql_query(sql, conn, params=params)

    @property
    def stations(self) -> pd.DataFrame:
        """Get list of available stations."""
        if not hasattr(self, '_stations_cache'):
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
            self._level_data = None
            self._discharge_data = None
            self._measured_data = None
            self._rating_curves = None

    @property
    def level_data(self) -> pd.DataFrame:
        """Get level timeseries data for current station."""
        if self._level_data is None:
            self._level_data = self._load_timeseries_data('level')
        return self._level_data

    @property
    def discharge_data(self) -> pd.DataFrame:
        """Get discharge timeseries data for current station."""
        if self._discharge_data is None:
            self._discharge_data = self._load_timeseries_data('discharge')
        return self._discharge_data

    @property
    def measured_data(self) -> pd.DataFrame:
        """Get measured discharge points for current station."""
        if self._measured_data is None:
            self._measured_data = self._load_measured_data()
        return self._measured_data

    @property
    def rating_curves(self) -> pd.DataFrame:
        """Get rating curve parameters for current station."""
        if self._rating_curves is None:
            self._rating_curves = self._load_rating_curves()
        return self._rating_curves

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

    def _load_measured_data(self) -> pd.DataFrame:
        """Load measured discharge points with date filtering."""
        if self._station_id is None:
            return pd.DataFrame()

        sql = """
        SELECT date, level, discharge, consistency
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

        sql += " ORDER BY date"

        df = self._read_sql(sql, tuple(params), parse_dates=['date'])
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
        ORDER BY start_date, segment_number
        """

        df = self._read_sql(sql, (self._station_id,), parse_dates=['start_date', 'end_date'])
        if not df.empty:
            # Handle NULL end_date
            df['end_date'] = df['end_date'].fillna(pd.Timestamp('2099-12-31'))
        return df

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

    def _calculate_max_measured_discharge_by_period(self) -> dict:
        """Calculate maximum measured discharge for each rating curve period."""
        max_discharge_by_period = {}

        for _, curve in self.rating_curves.iterrows():
            period_data = self.measured_data[
                (self.measured_data['date'] >= curve['start_date']) &
                (self.measured_data['date'] <= curve['end_date'])
            ]

            if not period_data.empty:
                max_discharge_by_period[curve['segment_number']] = period_data['discharge'].max()
            else:
                max_discharge_by_period[curve['segment_number']] = 0

        return max_discharge_by_period

    def _get_plot_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get plot date range based on filters or data."""
        if self._start_date and self._end_date:
            return pd.Timestamp(self._start_date), pd.Timestamp(self._end_date)

        # Find earliest and latest dates from all data sources
        dates = []
        for data in [self.level_data, self.discharge_data, self.measured_data]:
            if not data.empty:
                dates.extend([data['date'].min(), data['date'].max()])

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
        else:  # More than 5 years
            # YearLocator doesn't have interval parameter in older versions
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    def plot_timeseries(self, plot_type: str, figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """
        Create timeseries plot with rating curve indicators.

        Args:
            plot_type: 'Level' or 'Discharge'
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        if self._station_id is None:
            raise ValueError("No station selected. Use set_station() first.")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get appropriate data
        if plot_type == "Level":
            data = self.level_data
            value_col = 'level'
            ylabel = 'Water Level (cm)'
            color = 'b'
            label = 'Water Level'
        else:
            data = self.discharge_data
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
        title_text = f'Station {self._station_id} - {plot_type} Timeseries\\n'
        title_text += 'Red dashed lines: Rating curve periods | Orange dotted lines: Level limits'

        plt.title(title_text)
        plt.tight_layout()

        return fig

    def get_data_summary(self) -> dict:
        """Get summary statistics for current station data."""
        return {
            'level_records': len(self.level_data),
            'discharge_records': len(self.discharge_data),
            'measured_points': len(self.measured_data),
            'rating_curves': len(self.rating_curves)
        }


class DashboardApp:
    """Main Streamlit dashboard application."""

    def __init__(self, db_path: str = "data/hydrodata.sqlite"):
        self.analyzer = StationAnalyzer(db_path)

    def run(self):
        """Run the Streamlit dashboard."""
        st.title("🌊 Rating Curve Data Dashboard")
        st.markdown("Subtitle holder")

        # Sidebar controls
        st.sidebar.header("Controls")

        try:
            # Station selection
            stations = self.analyzer.stations

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

            # Plot type selection
            st.sidebar.subheader("Plot Type")
            plot_type = st.sidebar.radio(
                "Select data to plot:",
                options=["Level", "Discharge"],
                index=0
            )

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

            # Set station and load data
            with st.spinner("Loading data..."):
                self.analyzer.set_station(station_id, start_date, end_date)
                summary = self.analyzer.get_data_summary()

            # Display data summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Level Records", summary['level_records'])
            with col2:
                st.metric("Discharge Records", summary['discharge_records'])
            with col3:
                st.metric("Measured Points", summary['measured_points'])
            with col4:
                st.metric("Rating Curves", summary['rating_curves'])

            # Main plot
            data_available = (
                (plot_type == "Level" and summary['level_records'] > 0) or
                (plot_type == "Discharge" and summary['discharge_records'] > 0)
            )

            if data_available:
                st.subheader(f"{plot_type} Timeseries with Rating Curve Indicators")

                fig = self.analyzer.plot_timeseries(plot_type)
                st.pyplot(fig)

                # Show rating curve table
                if summary['rating_curves'] > 0:
                    st.subheader("Rating Curve Parameters")
                    display_curves = self.analyzer.rating_curves.copy()
                    display_curves['start_date'] = display_curves['start_date'].dt.strftime('%Y-%m-%d')
                    display_curves['end_date'] = display_curves['end_date'].dt.strftime('%Y-%m-%d')
                    display_curves['end_date'] = display_curves['end_date'].replace('2099-12-31', 'Current')

                    st.dataframe(display_curves, use_container_width=True)
            else:
                st.warning(f"No {plot_type.lower()} data found for station {station_id}")

        except Exception as e:
            st.error(f"Error loading data: {e}")


def main():
    """Entry point for the dashboard."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()