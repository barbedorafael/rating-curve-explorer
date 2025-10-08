import sys

sys.path.append('scripts')

from hydrodb import HydroDB
from rc_fit import RatingCurveFitter

def summarize_segments(df):

    print(f"Number of segments: {len(df)}")
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  Segment {i+1} ({row['segment_number']}): Q = {row['a_param']:.3f}×(H-{row['h0_param']:.3f})^{row['n_param']:.3f}")
        print(f"    Height range: {row['h_min']} - {row['h_max']} cm")
        print(f"    Validity: {row['start_date']} - {row['end_date']}")
    print()


db_path = "data/hydrodata.sqlite"
station_id = 87170000
station = HydroDB(db_path, station_id)

rcs = station.load_rating_curve_data()
dates = rcs['start_date'].unique()
current_date = dates[-1]
current_rc = rcs[rcs['start_date'] == current_date]
previous_date = dates[-2]
previous_rc = rcs[rcs['start_date'] == previous_date]
extrapolation_segments = rcs[rcs['segment_number'].str.split('/').str[0] == rcs['segment_number'].str.split('/').str[1]]

print(f"Rating Curve Adjuster for station {station_id}")
print("=" * 60)
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
    start_date='2023-08-20',
    end_date=current_rc.iloc[0].end_date)
print(f"Loaded {len(data['level'])} measurements")
print(f"Level range: {data['level'].min():.2f} - {data['level'].max():.2f} m")
print(f"Discharge range: {data['discharge'].min():.2f} - {data['discharge'].max():.2f} m³/s")


# ==============================================
# Fitter
# ==============================================

last_seg = current_rc.iloc[1]
extrapolation_params = {
    'a': last_seg.a_param,
    'x0': last_seg.h0_param,
    'n': last_seg.n_param,
    'x_start': last_seg.h_min
}

# init Fitter
datafit = data
fitter = RatingCurveFitter(
            datafit,
            x_min=1.2, 
            last_segment_params=extrapolation_params,
            fixed_breakpoints=[3.13]
            )

fitter.load_rcs(rcs.loc[rcs.start_date>=previous_date])

# Analyze current adjustments (raw)
plot_id = fitter.existing_curves[-1].__dict__['cid']
fitter.plot_results(plot_id, str(station_id))

# Fit new curve
result = fitter.fit_segments(
    n_segments=2, 
    # curve_crossing_weight=0
    )
print("\nNew adjusted rating curve...")
fitter.plot_results('new', str(station_id))

fitter.plot_curves()

for segment in result['segments']:
    print(segment.__dict__)