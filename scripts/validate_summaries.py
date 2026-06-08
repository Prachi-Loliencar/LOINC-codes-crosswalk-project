"""
Verify that all summary parquet files exist and contain
the expected columns and row counts. Runs in CI before deployment.
"""

import pandas as pd
import sys
from pathlib import Path

SUMMARY_DIR = Path("data/results/summary")

EXPECTED = {
    "primary_by_config.parquet": [
        "corpus_strategy",
        "model_desc",
        "n_distractors",
        "mrr_grouped",
        "n",
    ],
    "filter_by_coverage.parquet": [
        "filter_condition",
        "coverage_pattern",
        "has_method",
        "mrr_grouped",
        "n",
    ],
    "st_by_coverage.parquet": [
        "model_type",
        "strategy",
        "coverage_pattern",
        "mrr_grouped",
        "n",
    ],
    "st_by_noise.parquet": [
        "model_type",
        "strategy",
        "noise_level",
        "mrr_grouped",
        "n",
    ],
}

errors = []
for filename, required_cols in EXPECTED.items():
    path = SUMMARY_DIR / filename
    if not path.exists():
        errors.append(f"MISSING: {filename}")
        continue
    df = pd.read_parquet(path)
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"{filename}: missing columns {missing_cols}")
    if len(df) == 0:
        errors.append(f"{filename}: empty dataframe")

if errors:
    print("Validation failed:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)

print(f"All {len(EXPECTED)} summary files validated.")
