"""
CLI Entry point to execute feature engineering pipeline.
"""

import sys
import os
import argparse

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.temporal_pipeline import TemporalFeaturePipeline


def main():
    parser = argparse.ArgumentParser(description="Build Formula 1 Temporal Feature Store")
    parser.add_argument("--start_year", type=int, default=1995, help="Historical starting year")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    pipeline = TemporalFeaturePipeline(start_year=args.start_year)
    train_df, monza_df = pipeline.build_all_features(output_dir=args.output_dir)
    print(f"\n[SUCCESS] Feature build complete.")
    print(f"Training dataset shape: {train_df.shape}")
    print(f"2026 Monza target dataset shape: {monza_df.shape}")


if __name__ == "__main__":
    main()
