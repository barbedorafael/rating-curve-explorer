#!/usr/bin/env python3
"""
Test script for Johnson Rating Curve Adjuster
"""

import sys
import os
sys.path.append('scripts')

from rc_adjust_johnson import JohnsonRatingCurveAdjuster

# Test the implementation
if __name__ == "__main__":
    # Initialize adjuster
    db_path = "data/hydrodata.sqlite"
    station_id = 72810000  # Example station from CLAUDE.md

    print(f"Testing Johnson Rating Curve Adjuster for station {station_id}")
    print("=" * 60)

    try:
        adjuster = JohnsonRatingCurveAdjuster(db_path, station_id)

        # Load stage-discharge data
        print("Loading stage-discharge data...")
        data = adjuster.load_stage_discharge_data(start_date='2014-03-11', end_date='2021-12-31')
        print(f"Loaded {len(data['levels'])} measurements")
        print(f"Level range: {data['levels'].min():.1f} - {data['levels'].max():.1f} cm")
        print(f"Discharge range: {data['discharges'].min():.2f} - {data['discharges'].max():.2f} m³/s")

        # Test with 2 segments
        print("\nTesting optimization with 2 segments...")
        extrapolation_params = [137.7506, 0.55, 1.54]  # From data

        result = adjuster.optimize_segments(
            num_segments=3,
            extrapolation_params=extrapolation_params,
            level_range=[0.52, 1.1],
            max_iterations=100,  # Reduced for testing
            population_size=50   # Reduced for testing
        )

        if result['success'] or result['objective_value'] < 1000:
            print(f"Optimization completed with objective value: {result['objective_value']:.3f}")

            # Get summary
            summary = adjuster.get_optimization_summary()
            if summary:
                print(f"\nOptimization Summary:")
                print(f"Number of segments: {summary['num_segments']}")
                print(f"Max continuity error: {summary['max_continuity_error']:.3f}%")

                for i, segment in enumerate(summary['segments']):
                    print(f"Segment {i+1}: h=[{segment['h_min']:.0f}-{segment['h_max']:.0f}] cm, "
                          f"a={segment['a_param']:.4f}, h0={segment['h0_param']:.2f}, n={segment['n_param']:.3f}")
                adjuster.plot_rating_curve()
        else:
            print(f"Optimization failed with objective value: {result['objective_value']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()