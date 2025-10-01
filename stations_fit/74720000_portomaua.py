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
station_id = 74720000
station = HydroDB(db_path, station_id)

rcs = station.load_rating_curve_data()
last_rc = rcs[rcs['start_date'] == rcs['start_date'].max()]
extrapolation_segments = rcs[rcs['segment_number'].str.split('/').str[0] == rcs['segment_number'].str.split('/').str[1]]

print("Rating Curve Data:")
print(f"Total curves found: {len(rcs)}")
print()

print("Last Rating Curve (most recent):")
summarize_segments(last_rc)

print("Extrapolation Segments (highest range):")
summarize_segments(extrapolation_segments)

# Load stage-discharge data
print("Loading stage-discharge data...")
data = station.load_stage_discharge_data(start_date=last_rc.iloc[0].start_date, end_date=last_rc.iloc[0].end_date)
print(f"Loaded {len(data['level'])} measurements")
print(f"Level range: {data['level'].min():.2f} - {data['level'].max():.2f} m")
print(f"Discharge range: {data['discharge'].min():.2f} - {data['discharge'].max():.2f} m³/s")

x = data['level'].values
y = data['discharge'].values

# Create fitter
extrapolation_params = { # Only one RC here
    'a': extrapolation_segments.iloc[0].a_param,
    'x0': extrapolation_segments.iloc[0].h0_param,
    'n': extrapolation_segments.iloc[0].n_param,
    'x_start': extrapolation_segments.iloc[0].h_min
}
fitter = RatigCurveFitter(x, y, extrapolation_params)

# Create a mock result with existing segments for plotting
existing_segments = fitter.load_existing_segments(last_rc)
existing_result = {
    'segments': existing_segments,
    'n_segments': len(existing_segments),
}

print("\nPlotting existing rating curve...")
fitter.plot_results(existing_result, str(station_id))

# Existing result adjusts well...



