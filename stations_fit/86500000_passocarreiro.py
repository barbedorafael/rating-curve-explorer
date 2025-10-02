import sys

sys.path.append('scripts')

from hydrodb import HydroDB
from rc_fit import *

def summarize_segments(df):

    print(f"Number of segments: {len(df)}")
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  Segment {i+1} ({row['segment_number']}): Q = {row['a_param']:.3f}×(H-{row['h0_param']:.3f})^{row['n_param']:.3f}")
        print(f"    Height range: {row['h_min']} - {row['h_max']} cm")
        print(f"    Validity: {row['start_date']} - {row['end_date']}")
    print()


db_path = "data/hydrodata.sqlite"
station_id = 86500000
station = HydroDB(db_path, station_id)

print(f"Rating Curve Adjuster for station {station_id}")
print("=" * 60)

rcs = station.load_rating_curve_data()
current_date = rcs['start_date'].max()
current_rc = rcs[rcs['start_date'] == current_date]
# previous_date = '2010-05-19'              # Current = previous here
# previous_rc = rcs[rcs['start_date'] == previous_date]
extrapolation_segments = rcs[rcs['segment_number'].str.split('/').str[0] == rcs['segment_number'].str.split('/').str[1]]

print("Rating Curve Data:")
print(f"Total curves found: {len(rcs)}")
print()

print("Last Rating Curve (most recent):")
summarize_segments(current_rc)

print("Extrapolation Segments (highest range):")
summarize_segments(extrapolation_segments)

# Load stage-discharge data
print("Loading stage-discharge data...")
data = station.load_stage_discharge_data(
    start_date=current_rc.iloc[0].end_date,
    end_date='2025-12-31')
print(f"Loaded {len(data['level'])} measurements")
print(f"Level range: {data['level'].min():.2f} - {data['level'].max():.2f} m")
print(f"Discharge range: {data['discharge'].min():.2f} - {data['discharge'].max():.2f} m³/s")

x = data['level'].values
y = data['discharge'].values


extrapolation_params = { # Using the middle segment (just fixing the bottom part of the curve)
    'a': current_rc.iloc[-2].a_param,
    'x0': current_rc.iloc[-2].h0_param,
    'n': current_rc.iloc[-2].n_param,
    'x_start': current_rc.iloc[-2].h_min/100
}

# init Fitter
fitter = RatigCurveFitter(x, y, x_max=3.0, last_segment_params=extrapolation_params, fixed_breakpoints=[1.6])

# Analyze current adjustment (raw)
existing_segments = fitter.load_existing_segments(current_rc)
existing_result = {
    'segments': existing_segments,
    'n_segments': len(existing_segments),
}

print("\nPlotting current rating curve...")
fitter.plot_results(existing_result, str(station_id))

# Fit new curve for entire period
result = fitter.fit_segments(n_segments=2)
print("\nNew adjusted rating curve...")
fitter.plot_results(result, str(station_id))