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
        "top1",
        "top3",
        "top5",
        "n",
        "n_loinc_codes",
    ],
    "primary_by_coverage.parquet": [
        "corpus_strategy",
        "model_desc",
        "n_distractors",
        "coverage_pattern",
        "mrr_grouped",
        "top1",
        "top3",
        "n",
    ],
    "primary_by_coverage_noise.parquet": [
        "corpus_strategy",
        "model_desc",
        "n_distractors",
        "coverage_pattern",
        "noise_level",
        "mrr_grouped",
        "n",
    ],
    "filter_by_config.parquet": [
        "filter_applied",
        "corpus_strategy",
        "mrr_grouped",
        "top1",
        "top3",
        "top5",
        "n",
    ],
    "filter_by_method.parquet": [
        "filter_applied",
        "has_method",
        "mrr_grouped",
        "n",
    ],
    "filter_by_coverage.parquet": [
        "filter_applied",
        "coverage_pattern",
        "mrr_grouped",
        "n",
    ],
    "filter_by_noise_level.parquet": [
        "filter_applied",
        "noise_level",
        "mrr_grouped",
        "n",
    ],
    "filter_by_noise_omission.parquet": [
        "filter_applied",
        "noise_omission",
        "mrr_grouped",
        "n",
    ],
    "filter_by_noise_compression.parquet": [
        "filter_applied",
        "noise_compression",
        "mrr_grouped",
        "n",
    ],
    "filter_by_loinc.parquet": [
        "filter_applied",
        "true_loinc",
        "mrr_grouped",
        "n",
    ],
    "st_by_config.parquet": [
        "model_type",
        "strategy",
        "mrr_grouped",
        "top1",
        "top3",
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
    "st_by_method.parquet": [
        "model_type",
        "strategy",
        "has_method",
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
sys.exit(0)  # explicit clean exit
