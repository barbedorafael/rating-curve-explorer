#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Rating Curve Data Cleaning

Features:
- Station filtering
- Combined level/discharge timeseries with rating curve validity indicators
- Measured discharge points overlay
- Extrapolation region highlighting
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Rating Curve Explorer",
    page_icon="🌊",
    layout="wide"
)

def get_db_connection(db_path="data/hydrodata.sqlite"):
    """Get SQLite database connection."""
    conn = sqlite3.connect(db_path)
    return conn

@st.cache_data
def get_stations(db_path="data/hydrodata.sqlite"):
    """Get list of available stations."""
    conn = get_db_connection(db_path)
    query = """
    SELECT DISTINCT s.station_id, s.name, s.river_id
    FROM stations s
    WHERE s.station_id IN (
        SELECT DISTINCT station_id FROM timeseries_cota
        UNION
        SELECT DISTINCT station_id FROM rating_curve
    )
    ORDER BY s.station_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data
def get_timeseries_data(station_id, start_date=None, end_date=None, db_path="data/hydrodata.sqlite"):
    """Get level and discharge timeseries data for a station."""
    conn = get_db_connection(db_path)

    # Build date filter
    date_filter = ""
    params = [station_id]

    if start_date:
        date_filter += " AND date >= ?"
        params.append(start_date.strftime('%Y-%m-%d'))
    if end_date:
        date_filter += " AND date <= ?"
        params.append(end_date.strftime('%Y-%m-%d'))

    # Get level data
    level_query = f"""
    SELECT date, value as level, status
    FROM timeseries_cota
    WHERE station_id = ? {date_filter}
    ORDER BY date
    """
    level_data = pd.read_sql_query(level_query, conn, params=params)
    if not level_data.empty:
        level_data['date'] = pd.to_datetime(level_data['date'])

    # Get discharge data
    discharge_query = f"""
    SELECT date, value as discharge, status
    FROM timeseries_vazao
    WHERE station_id = ? {date_filter}
    ORDER BY date
    """
    discharge_data = pd.read_sql_query(discharge_query, conn, params=params)
    if not discharge_data.empty:
        discharge_data['date'] = pd.to_datetime(discharge_data['date'])

    conn.close()
    return level_data, discharge_data

def fill_missing_dates(data, date_col, value_col):
    """Fill missing dates with NaN to prevent interpolation."""
    if data.empty:
        return data

    # Create complete date range
    date_range = pd.date_range(start=data[date_col].min(), end=data[date_col].max(), freq='D')

    # Create DataFrame with all dates
    complete_df = pd.DataFrame({date_col: date_range})

    # Merge with original data
    filled_data = complete_df.merge(data, on=date_col, how='left')

    return filled_data

@st.cache_data
def get_measured_discharge(station_id, start_date=None, end_date=None, db_path="data/hydrodata.sqlite"):
    """Get measured discharge points for a station."""
    conn = get_db_connection(db_path)

    date_filter = ""
    params = [station_id]

    if start_date:
        date_filter += " AND date >= ?"
        params.append(start_date.strftime('%Y-%m-%d'))
    if end_date:
        date_filter += " AND date <= ?"
        params.append(end_date.strftime('%Y-%m-%d'))

    query = f"""
    SELECT date, level, discharge, consistency
    FROM stage_discharge
    WHERE station_id = ? {date_filter}
    AND level IS NOT NULL AND discharge IS NOT NULL
    ORDER BY date
    """

    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])

    conn.close()
    return df

@st.cache_data
def get_rating_curves(station_id, db_path="data/hydrodata.sqlite"):
    """Get rating curve parameters and validity periods for a station."""
    conn = get_db_connection(db_path)

    query = """
    SELECT segment_number, start_date, end_date, h_min, h_max,
           h0_param, a_param, n_param
    FROM rating_curve
    WHERE station_id = ?
    ORDER BY start_date, segment_number
    """

    df = pd.read_sql_query(query, conn, params=[station_id])
    if not df.empty:
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'].fillna('2099-12-31'))

    conn.close()
    return df

def calculate_max_measured_discharge_by_period(measured_data, rating_curves):
    """Calculate maximum measured discharge for each rating curve period."""
    max_discharge_by_period = {}

    for _, curve in rating_curves.iterrows():
        period_data = measured_data[
            (measured_data['date'] >= curve['start_date']) &
            (measured_data['date'] <= curve['end_date'])
        ]

        if not period_data.empty:
            max_discharge_by_period[curve['segment_number']] = period_data['discharge'].max()
        else:
            max_discharge_by_period[curve['segment_number']] = 0

    return max_discharge_by_period

def plot_timeseries_with_indicators(data, measured_data, rating_curves, station_id, plot_type, start_date=None, end_date=None):
    """Create single timeseries plot with rating curve indicators."""

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    # Calculate max measured discharge by period for extrapolation detection
    max_discharge_by_period = calculate_max_measured_discharge_by_period(measured_data, rating_curves)

    # Filter data by date range for proper axis scaling
    if start_date and end_date:
        plot_start = pd.Timestamp(start_date)
        plot_end = pd.Timestamp(end_date)
    elif not data.empty:
        plot_start = data['date'].min()
        plot_end = data['date'].max()
    else:
        plot_start = pd.Timestamp('1960-01-01')
        plot_end = pd.Timestamp('2025-01-01')

    # Fill missing dates to prevent interpolation
    if not data.empty:
        if plot_type == "Level":
            filled_data = fill_missing_dates(data, 'date', 'level')
        else:
            filled_data = fill_missing_dates(data, 'date', 'discharge')
    else:
        filled_data = data

    # Track legend labels to avoid duplicates
    legend_labels = set()

    # Plot main timeseries data
    if not filled_data.empty:
        if plot_type == "Level":
            # Plot level data
            ax.plot(filled_data['date'], filled_data['level'], 'b-', linewidth=1, alpha=0.7, label='Water Level')
            ax.set_ylabel('Water Level (cm)', color='b')
            ax.tick_params(axis='y', labelcolor='b')
        else:
            # Plot discharge data - simple clean line like level
            ax.plot(filled_data['date'], filled_data['discharge'], 'g-', linewidth=1, alpha=0.7, label='Discharge')
            ax.set_ylabel('Discharge (m³/s)', color='g')
            ax.tick_params(axis='y', labelcolor='g')

    # Add rating curve validity indicators
    for _, curve in rating_curves.iterrows():
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

            ax.hlines(curve['h_min'], line_start, line_end, colors='orange', linestyles=':', alpha=0.5, linewidth=1)
            ax.hlines(curve['h_max'], line_start, line_end, colors='orange', linestyles=':', alpha=0.5, linewidth=1)

            # Add text annotation for segment below the line
            mid_date = line_start + (line_end - line_start) / 2
            y_pos = curve['h_min']
            ax.text(mid_date, y_pos, f"Seg {curve['segment_number']}",
                   rotation=0, fontsize=8, ha='center', va='bottom', color='red')

    # Add measured points (SAME FOR BOTH PLOTS)
    if not measured_data.empty:
        # Color by consistency
        good_points = measured_data[measured_data['consistency'] == 1]
        poor_points = measured_data[measured_data['consistency'] != 1]

        if plot_type == "Level":
            y_values_good = good_points['level']
            y_values_poor = poor_points['level']
        else:
            y_values_good = good_points['discharge']
            y_values_poor = poor_points['discharge']

        if not good_points.empty:
            ax.scatter(good_points['date'], y_values_good,
                      c='darkgreen', s=30, alpha=0.8, label='Measurement Expeditions', zorder=5)
            legend_labels.add('Measurement Expeditions')

        if not poor_points.empty:
            ax.scatter(poor_points['date'], y_values_poor,
                      c='orange', s=30, alpha=0.8, label='Measurement Expeditions (Poor)', zorder=5)
            legend_labels.add('Measurement Expeditions')

    # Set proper axis limits based on filtered data
    ax.set_xlim(plot_start, plot_end)

    ax.set_xlabel('Date')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Add legend
    ax.legend(loc='upper left')

    # Add grid
    ax.grid(True, alpha=0.3)

    title_text = f'Station {station_id} - {plot_type} Timeseries\n'
    title_text += 'Red dashed lines: Rating curve periods | Orange dotted lines: Level limits'

    plt.title(title_text)
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit app."""

    st.title("🌊 Rating Curve Data Dashboard")
    st.markdown("Interactive tool for visualizing and cleaning hydrological data")

    # Sidebar for controls
    st.sidebar.header("Controls")

    # Load stations
    try:
        stations = get_stations()

        if stations.empty:
            st.error("No stations found in database")
            return

        # Station selection
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
            start_date = st.date_input("Start Date", value=None, min_value=datetime(1900, 1, 1), max_value=datetime(2099, 12, 31))
        with col2:
            end_date = st.date_input("End Date", value=None, min_value=datetime(1900, 1, 1), max_value=datetime(2099, 12, 31))

        # Load data for selected station
        with st.spinner("Loading data..."):
            level_data, discharge_data = get_timeseries_data(station_id, start_date, end_date)
            measured_data = get_measured_discharge(station_id, start_date, end_date)
            rating_curves = get_rating_curves(station_id)

        # Display data summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Level Records", len(level_data))
        with col2:
            st.metric("Discharge Records", len(discharge_data))
        with col3:
            st.metric("Measured Points", len(measured_data))
        with col4:
            st.metric("Rating Curves", len(rating_curves))

        # Select appropriate data for plotting
        if plot_type == "Level":
            plot_data = level_data
        else:
            plot_data = discharge_data

        # Main plot
        if not plot_data.empty:
            st.subheader(f"{plot_type} Timeseries with Rating Curve Indicators")

            fig = plot_timeseries_with_indicators(
                plot_data, measured_data, rating_curves, station_id, plot_type, start_date, end_date
            )

            st.pyplot(fig)

            # Show rating curve table
            if not rating_curves.empty:
                st.subheader("Rating Curve Parameters")
                display_curves = rating_curves.copy()
                display_curves['start_date'] = display_curves['start_date'].dt.strftime('%Y-%m-%d')
                display_curves['end_date'] = display_curves['end_date'].dt.strftime('%Y-%m-%d')
                display_curves['end_date'] = display_curves['end_date'].replace('2099-12-31', 'Current')

                st.dataframe(display_curves, use_container_width=True)

        else:
            st.warning(f"No data found for station {station_id}")

    except Exception as e:
        st.error(f"Error loading data: {e}")

if __name__ == "__main__":
    main()