from scripts.hydrodb import HydroDB

db = HydroDB("data/hydrodata.sqlite")

# Timeseries
db.plot_timeseries(station=74100000, variable="level", start_date="2024-01-01", end_date="2025-01-01", rolling=7)
db.plot_timeseries(station=74100000, variable="discharge", rolling=14)

# Vertical profile
db.plot_vertical_profile(survey_id=3875)

# Rating curve (segmented by level)
db.plot_rating_curve(
    station=74100000,
    start_date="2020-07-09",
    end_date="2025-12-31",
    level_breaks=[100, 363, 3000],   # H segments: [min,H1], (H1,H2], (H2,H3]
    fit=True
)
