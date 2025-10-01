import sys

sys.path.append('scripts')

from hydrodb import HydroDB
from rc_fit import RatigCurveFitter

def summarize_segments(df):

    print(f"Number of segments: {len(df)}")
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  Segment {i+1} ({row['segment_number']}): Q = {row['a_param']:.3f}×(H-{row['h0_param']:.3f})^{row['n_param']:.3f}")
        print(f"    Height range: {row['h_min']} - {row['h_max']} cm")
        print(f"    Validity: {row['start_date']} - {row['end_date']}")
    print()


    print(f"Rating Curve Adjuster for station {station_id}")
    print("=" * 60)

db_path = "data/hydrodata.sqlite"
station_id = 86580000
station = HydroDB(db_path, station_id)

rcs = station.load_rating_curve_data()
current_date = rcs['start_date'].max()
current_rc = rcs[rcs['start_date'] == current_date]
previous_date = '2013-08-26'
previous_rc = rcs[rcs['start_date'] == previous_date]
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
    start_date=previous_date,
    end_date='2025-12-31')
print(f"Loaded {len(data['level'])} measurements")
print(f"Level range: {data['level'].min():.2f} - {data['level'].max():.2f} m")
print(f"Discharge range: {data['discharge'].min():.2f} - {data['discharge'].max():.2f} m³/s")

x = data['level'].values
y = data['discharge'].values


extrapolation_params = { # Second to last extrapolation segment
    'a': extrapolation_segments.iloc[-2].a_param,
    'x0': extrapolation_segments.iloc[-2].h0_param,
    'n': extrapolation_segments.iloc[-2].n_param,
    'x_start': extrapolation_segments.iloc[-2].h_min
}

# init Fitter
fitter = RatigCurveFitter(x, y, extrapolation_params)

# Analyze current adjustment (raw)
existing_segments = fitter.load_existing_segments(current_rc)
existing_result = {
    'segments': existing_segments,
    'n_segments': len(existing_segments),
}

print("\nPlotting current rating curve...")
fitter.plot_results(existing_result, str(station_id))

# Analyze previous adjustment (consisted)
existing_segments = fitter.load_existing_segments(previous_rc)
existing_result = {
    'segments': existing_segments,
    'n_segments': len(existing_segments),
}

print("\nPlotting previous rating curve...")
fitter.plot_results(existing_result, str(station_id))

