import sys

sys.path.append('scripts')
sys.path.append('scripts/rc_adjust')

from hydrodb import HydroDB
from rc_fit import SegmentedPowerCurveFitter

# Test the implementation
if __name__ == "__main__":
    # Initialize adjuster
    db_path = "data/hydrodata.sqlite"
    station_id = 72810000  # Example station from CLAUDE.md

    print(f"Testing Johnson Rating Curve Adjuster for station {station_id}")
    print("=" * 60)

    station = HydroDB(db_path, station_id)

    # Load stage-discharge data
    print("Loading stage-discharge data...")
    data = station.load_stage_discharge_data(start_date='2014-11-08', end_date='2021-12-31')
    print(f"Loaded {len(data['level'])} measurements")
    print(f"Level range: {data['level'].min():.2f} - {data['level'].max():.2f} m")
    print(f"Discharge range: {data['discharge'].min():.2f} - {data['discharge'].max():.2f} m³/s")

    # Define last segment: y = a * (x - x0)^n
    last_segment_params = {
        'a': 137.7506,      # Coefficient
        'x0': 0.55,         # X offset
        'n': 1.54,          # Power
        'x_start': 1.2      # Where this segment starts
    }

    x = data['level'].values
    y = data['discharge'].values
    # Create fitter with last segment
    fitter = SegmentedPowerCurveFitter(x, y)#, last_segment_params)

    result = fitter.fit_segments(n_segments=3)

    fitter.plot_results(result)


