# ablation.py
# Retrieval ablation runner.

import itertools
import pandas as pd
import numpy as np
import logging


from src.clinical_utils import clean_text
from src.model_building_utils import (
    PipelineConfig,
    expand_loinc_lcn,
    normalize_elr,
    assign_splits,
    evaluate_pipeline,
    compute_relatednames_stopwords,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("logs/ablation.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------


def load_data():
    """Loads and preprocesses all inputs. Returns (elr, loinc, distractors, rn_stopwords)."""
    elr = pd.read_csv("data/processed/elr_simulated.csv")
    loinc = pd.read_csv("data/processed/covid_surveillance_loinc.csv")
    loinc = loinc[~loinc.method_typ.isna()].copy()

    loinc["expanded_lcn"] = (
        loinc["long_common_name"].map(clean_text).map(expand_loinc_lcn)
    )
    elr["elr_name_normalized"] = elr["elr_name"].map(clean_text).map(normalize_elr)
    elr = assign_splits(elr)

    df_distractors = pd.read_csv("data/processed/distractor_loincs.csv")
    df_distractors["expanded_lcn"] = (
        df_distractors["long_common_name"].map(clean_text).map(expand_loinc_lcn)
    )
    # Strictly non-COVID respiratory codes only — COVID-adjacent codes fail to
    # raise SARSCOV2 IDF because they also contain SARSCOV2 tokens.
    df_distractors_noncovid = df_distractors[
        ~df_distractors["long_common_name"].str.contains(
            r"SARS|COVID|coronavirus", case=False, na=False, regex=True
        )
    ].copy()

    rn_stopwords = compute_relatednames_stopwords(loinc, threshold=0.85)

    return elr, loinc, df_distractors_noncovid, rn_stopwords


# ---------------------------------------------------------------------------
# Primary ablation
# Q1: corpus strategy  Q2: distractor count
# Establishes TF-IDF retrieval baseline.
# ---------------------------------------------------------------------------

PRIMARY_CORPUS_STRATEGIES = [
    "lcn_only",
    "combined",
    "lcn_method_dict_combined",
    "lcn_filtered_rn_combined",
    "lcn_method_dict_filtered_rn",
    "component_weighted_method_dict",
]

PRIMARY_MODEL_CONFIGS = [
    ("tfidf_word", (1, 1)),
    ("tfidf_word", (1, 2)),
    ("tfidf_char", (3, 5)),
    ("tfidf_mixed", None),
]

PRIMARY_N_DISTRACTORS = [0, 143]


def run_primary(elr, loinc, df_distractors_noncovid, rn_stopwords) -> pd.DataFrame:
    """
    Primary ablation: corpus strategy x distractor count x model type.
    Validation set only. No post-retrieval filtering.
    Returns a flat DataFrame of per-row results.
    """
    val_df = elr[elr["split"] == "val"].copy()
    all_results = []

    total = (
        len(PRIMARY_N_DISTRACTORS)
        * len(PRIMARY_CORPUS_STRATEGIES)
        * len(PRIMARY_MODEL_CONFIGS)
    )
    print(f"Primary ablation: {total} runs")

    for n in PRIMARY_N_DISTRACTORS:
        distractors = (
            None if n == 0 else df_distractors_noncovid.sample(n, random_state=42)
        )
        for cs in PRIMARY_CORPUS_STRATEGIES:
            for model_type, ngram_range in PRIMARY_MODEL_CONFIGS:
                config = PipelineConfig(
                    filter_strategy="none",
                    corpus_strategy=cs,
                    model_type=model_type,
                    n_neighbors=6,
                    extended_corpus=(n > 0),
                    n_distractors=n,
                )
                results = evaluate_pipeline(
                    config=config,
                    df_loinc=loinc,
                    df_test=val_df,
                    ngram_range=ngram_range,
                    df_loinc_distractors=distractors,
                    rn_stopwords=rn_stopwords,
                )
                results["n_distractors"] = n
                results["model_desc"] = f"{model_type}_{ngram_range}"
                all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------------
# Secondary ablation
# Best corpus strategies, mixed model ngram/alpha sweep.
# ---------------------------------------------------------------------------

SECONDARY_CORPUS_STRATEGIES = [
    "lcn_method_dict_filtered_rn",
    "lcn_method_dict_combined",
    "component_weighted_method_dict",
    "lcn_filtered_rn_combined",
]

SECONDARY_N_DISTRACTORS = [0, 50, 100, 143]

SECONDARY_NON_MIXED = [
    ("tfidf_word", (1, 1), None, None, None),
    ("tfidf_char", (3, 5), None, None, None),
]

MIXED_NGRAM_COMBOS = [((1, 1), (3, 5)), ((1, 2), (3, 6))]
MIXED_ALPHAS = [0.3, 0.5, 0.7, 0.9]

SECONDARY_MIXED = [
    ("tfidf_mixed", None, ngram_word, ngram_char, alpha)
    for (ngram_word, ngram_char), alpha in itertools.product(
        MIXED_NGRAM_COMBOS, MIXED_ALPHAS
    )
]

SECONDARY_ALL_MODELS = SECONDARY_NON_MIXED + SECONDARY_MIXED


def run_secondary(elr, loinc, df_distractors_noncovid, rn_stopwords) -> pd.DataFrame:
    """
    Secondary ablation: best corpus strategies x mixed model ngram/alpha sweep.
    No post-retrieval filtering.
    Returns a flat DataFrame of per-row results.
    """
    val_df = elr[elr["split"] == "val"].copy()
    all_results = []

    total = (
        len(SECONDARY_N_DISTRACTORS)
        * len(SECONDARY_CORPUS_STRATEGIES)
        * len(SECONDARY_ALL_MODELS)
    )
    print(f"Secondary ablation: {total} runs")

    for n in SECONDARY_N_DISTRACTORS:
        distractors = (
            None if n == 0 else df_distractors_noncovid.sample(n, random_state=42)
        )
        for cs in SECONDARY_CORPUS_STRATEGIES:
            for (
                model_type,
                ngram_range,
                ngram_word,
                ngram_char,
                alpha,
            ) in SECONDARY_ALL_MODELS:
                config = PipelineConfig(
                    filter_strategy="none",
                    corpus_strategy=cs,
                    model_type=model_type,
                    n_neighbors=6,
                    extended_corpus=(n > 0),
                    n_distractors=n,
                    ngram_word=ngram_word if ngram_word else (1, 1),
                    ngram_char=ngram_char if ngram_char else (3, 5),
                    alpha=alpha if alpha else 0.5,
                )
                results = evaluate_pipeline(
                    config=config,
                    df_loinc=loinc,
                    df_test=val_df,
                    ngram_range=ngram_range,
                    df_loinc_distractors=distractors,
                    rn_stopwords=rn_stopwords,
                )
                results["n_distractors"] = n
                results["model_desc"] = (
                    f"mixed_w{ngram_word}_c{ngram_char}_a{alpha}"
                    if model_type == "tfidf_mixed"
                    else f"{model_type}_{ngram_range}"
                )
                all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------------
# Filter ablation
# Runs on the best corpus+model config from secondary ablation.
# Three conditions per row: no filter, oracle filter, brand filter.
# Stratified by coverage_pattern and has_method to isolate the gain.
# ---------------------------------------------------------------------------


def run_filter_ablation(
    elr,
    loinc,
    df_distractors_noncovid,
    rn_stopwords,
    best_corpus_strategy: str = "lcn_method_dict_combined",
    best_model_type: str = "tfidf_word",
    best_ngram_range: tuple = (1, 1),
    n_distractors: int = 0,
) -> pd.DataFrame:
    """
    Filter ablation: compares no-filter, oracle-filter, and brand-filter on the
    best config from secondary ablation.

    Designed to answer:
      - Oracle: how much does perfect metadata help? (upper bound)
      - Brand:  how much of that upper bound does brand imputation recover?

    Key stratification axes in the returned DataFrame:
      coverage_pattern, has_method, has_model

    Run on validation set. Pass best_* arguments from secondary ablation summary.
    """
    val_df = elr[elr["split"] == "val"].copy()
    distractors = (
        None
        if n_distractors == 0
        else df_distractors_noncovid.sample(n_distractors, random_state=42)
    )

    base_kwargs = dict(
        filter_strategy="none",
        corpus_strategy=best_corpus_strategy,
        model_type=best_model_type,
        n_neighbors=6,
        extended_corpus=(n_distractors > 0),
        n_distractors=n_distractors,
    )

    filter_conditions = [
        # (label, oracle_filter, brand_filter)
        ("no_filter", False, False),
        ("oracle_filter", True, False),
        ("brand_filter", False, True),
    ]

    all_results = []
    for label, oracle, brand in filter_conditions:
        config = PipelineConfig(**base_kwargs, oracle_filter=oracle, brand_filter=brand)
        results = evaluate_pipeline(
            config=config,
            df_loinc=loinc,
            df_test=val_df,
            ngram_range=best_ngram_range,
            df_loinc_distractors=distractors,
            rn_stopwords=rn_stopwords,
        )

        results["filter_condition"] = label

        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------------
# Single strategy testing - returns results and confidence scores for a single strategy
# ---------------------------------------------------------------------------


def assess_confidence(row, medium_thresh, high_thresh):
    top_score = row["top_score"]

    tier = (
        "HIGH"
        if top_score >= high_thresh
        else "MEDIUM"
        if top_score >= medium_thresh
        else "LOW"
    )

    return {
        "confidence_tier": tier,
        "manual_review": tier == "LOW",
    }


def run_single_strategy(
    elr,
    loinc,
    df_distractors_noncovid,
    rn_stopwords,
    best_corpus_strategy: str = "lcn_method_dict_combined",
    best_model_type: str = "tfidf_word",
    ngram_word: tuple = (1, 1),
    ngram_char: tuple = (3, 5),
    n_distractors: int = 0,
    eval_set: str = "val",
    alpha: float = 0.5,
) -> pd.DataFrame:
    config = PipelineConfig(
        filter_strategy="none",
        corpus_strategy=best_corpus_strategy,
        model_type=best_model_type,
        n_neighbors=6,
        extended_corpus=(n_distractors > 0),
        n_distractors=n_distractors,
        ngram_word=ngram_word,
        ngram_char=ngram_char,
        alpha=alpha,
    )
    results = evaluate_pipeline(
        config=config,
        df_loinc=loinc,
        df_test=elr[elr.split == "val"],
        ngram_range=ngram_word,
        df_loinc_distractors=df_distractors_noncovid,
        rn_stopwords=rn_stopwords,
    )

    # Compute on val set, apply to test
    q33 = results["top_score"].quantile(0.33)
    q67 = results["top_score"].quantile(0.67)

    if eval_set == "test":
        results = evaluate_pipeline(
            config=config,
            df_loinc=loinc,
            df_test=elr[elr.split == "test"],
            ngram_range=ngram_word,
            df_loinc_distractors=df_distractors_noncovid,
            rn_stopwords=rn_stopwords,
        )

    results["n_distractors"] = n_distractors
    results[["confidence_tier", "manual_review"]] = results.apply(
        lambda x: assess_confidence(x, q33, q67), axis=1, result_type="expand"
    )
    return results


# ---------------------------------------------------------------------------
# Convenience summaries (used by notebook)
# ---------------------------------------------------------------------------


def summarize_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Overall MRR by corpus x model x n_distractors."""
    return (
        df.groupby(["corpus_strategy", "model_desc", "n_distractors"])
        .mrr_grouped.mean()
        .unstack("n_distractors")
        .sort_values(0, ascending=False)
    )


def summarize_by_coverage(
    df: pd.DataFrame, mrr_col: str = "mrr_grouped"
) -> pd.DataFrame:
    """MRR by coverage_pattern for a results DataFrame."""
    return (
        df[df.n_distractors == 0]
        .groupby(["corpus_strategy", "model_desc", "coverage_pattern"])[mrr_col]
        .mean()
        .unstack("coverage_pattern")
    )


def summarize_filter_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter ablation summary.
    Shows grouped MRR by filter condition, coverage_pattern, and has_method.
    The has_method=0 sub-table is the primary finding.
    """
    return (
        df.groupby(["filter_condition", "coverage_pattern", "has_method"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            top1=("top1", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
    )


def brand_recovery_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes brand filter recovery fraction relative to oracle upper bound.
    Population restricted to has_method=0, has_model=1.
    """
    pop = df[(df["has_model"] == 1) & (df["has_method"] == 0)].copy()
    summary = (
        pop.groupby(["filter_condition", "coverage_pattern"])
        .mrr_grouped.mean()
        .unstack("filter_condition")
    )
    if "oracle_filter" in summary.columns and "brand_filter" in summary.columns:
        summary["brand_recovery_pct"] = (
            (summary["brand_filter"] - summary.get("no_filter", 0))
            / (summary["oracle_filter"] - summary.get("no_filter", 1e-9))
        ).clip(0, 1) * 100
    return summary


def save_streamlit_summaries(
    df_primary: pd.DataFrame,
    df_filter: pd.DataFrame,
    out_dir: str = "data/results/summary",
) -> None:
    """
    Generates all parquet summary files consumed by the Streamlit app.
    Call this at the bottom of ablation.py after the three ablation runs
    complete, passing the full row-level result DataFrames.

    Parameters
    ----------
    df_primary : output of run_primary()          — row-per-ELR results
    df_filter  : output of run_filter_ablation()  — row-per-ELR results
                 Must contain column filter_applied (not filter_condition).
    out_dir    : destination folder (created if absent)

    Output files and their exact column sets
    -----------------------------------------
    PRIMARY ABLATION
      primary_by_config.parquet
          corpus_strategy, model_desc, n_distractors,
          mrr_grouped, top1, top3, top5, n, n_loinc_codes

      primary_by_coverage.parquet
          corpus_strategy, model_desc, n_distractors, coverage_pattern,
          mrr_grouped, top1, top3, n

      primary_by_coverage_noise.parquet          ← needed by Tab 3 heatmap
          corpus_strategy, model_desc, n_distractors,
          coverage_pattern, noise_level, mrr_grouped, n

    FILTER ABLATION
      filter_by_config.parquet
          filter_applied, corpus_strategy, mrr_grouped, top1, top3, top5, n

      filter_by_method.parquet
          filter_applied, has_method, mrr_grouped, n

      filter_by_coverage.parquet
          filter_applied, coverage_pattern, mrr_grouped, n

      filter_by_noise_level.parquet
          filter_applied, noise_level, mrr_grouped, n

      filter_by_noise_omission.parquet
          filter_applied, noise_omission, mrr_grouped, n

      filter_by_noise_compression.parquet
          filter_applied, noise_compression, mrr_grouped, n

      filter_by_noise_compression.parquet
          filter_applied, noise_compression, mrr_grouped, n

      filter_by_loinc.parquet                    ← needed by Tab 7 scatter
          filter_applied, true_loinc, mrr_grouped, n
    """
    import os
    import pathlib

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # PRIMARY ABLATION
    # ------------------------------------------------------------------

    n_loinc = df_primary["true_loinc"].nunique()

    # Config-level: one row per (corpus_strategy, model_desc, n_distractors)
    # n_loinc_codes attached as a constant column for Tab 1 overview metric.
    (
        df_primary.groupby(["corpus_strategy", "model_desc", "n_distractors"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .assign(n_loinc_codes=n_loinc)
        .to_parquet(f"{out_dir}/primary_by_config.parquet", index=False)
    )

    # Coverage-pattern-level: Tab 3 coverage bar chart and table
    (
        df_primary.groupby(
            ["corpus_strategy", "model_desc", "n_distractors", "coverage_pattern"]
        )
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/primary_by_coverage.parquet", index=False)
    )

    # Coverage × noise_level: Tab 3 heatmap
    (
        df_primary.groupby(
            [
                "corpus_strategy",
                "model_desc",
                "n_distractors",
                "coverage_pattern",
                "noise_level",
            ]
        )
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/primary_by_coverage_noise.parquet", index=False)
    )

    # ------------------------------------------------------------------
    # FILTER ABLATION
    # ------------------------------------------------------------------

    # Config-level: one row per filter_applied condition
    # corpus_strategy kept so load_filter() can add corpus_label
    (
        df_filter.groupby(["filter_applied", "corpus_strategy"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/filter_by_config.parquet", index=False)
    )

    # filter × has_method: Tab 4 has_method bar chart
    (
        df_filter.groupby(["filter_applied", "has_method"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/filter_by_method.parquet", index=False)
    )

    # filter × coverage_pattern: Tab 4 coverage chart, Tab 6 TF-IDF baseline,
    # Tab 7 val-side per-pattern table
    (
        df_filter.groupby(["filter_applied", "coverage_pattern"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/filter_by_coverage.parquet", index=False)
    )

    # filter × noise dimensions: Tab 7 section C val-side noise breakdown
    # and _build_tfidf_ref() noise dict
    for noise_col in [
        "noise_level",
        "noise_omission",
        "noise_compression",
        "noise_corruption",
    ]:
        (
            df_filter.groupby(["filter_applied", noise_col])
            .agg(
                mrr_grouped=("mrr_grouped", "mean"),
                n=("mrr_grouped", "count"),
            )
            .reset_index()
            .to_parquet(f"{out_dir}/filter_by_{noise_col}.parquet", index=False)
        )

    # filter × true_loinc: Tab 7 section D per-LOINC scatter (val side)
    (
        df_filter.groupby(["filter_applied", "true_loinc"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/filter_by_loinc.parquet", index=False)
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    total_kb = (
        sum(p.stat().st_size for p in pathlib.Path(out_dir).glob("*.parquet")) / 1024
    )
    n_files = len(list(pathlib.Path(out_dir).glob("*.parquet")))
    print(f"Saved {n_files} summary parquet files to {out_dir}/")
    print(f"Total size: {total_kb:.1f} KB  (vs hundreds of MB for raw CSVs)")


# ---------------------------------------------------------------------------
# Entry point — runs all ablations and saves results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    elr, loinc, df_distractors_noncovid, rn_stopwords = load_data()

    print("\n=== Primary ablation ===")
    df_primary = run_primary(elr, loinc, df_distractors_noncovid, rn_stopwords)
    df_primary.to_csv("data/results/primary_ablation.csv", index=False)
    print(summarize_primary(df_primary).to_string())

    print("\n=== Secondary ablation ===")
    df_secondary = run_secondary(elr, loinc, df_distractors_noncovid, rn_stopwords)
    df_secondary.to_csv("data/results/secondary_ablation.csv", index=False)

    print("\n=== Filter ablation ===")
    df_filter = run_filter_ablation(elr, loinc, df_distractors_noncovid, rn_stopwords)
    df_filter.to_csv("data/results/filter_ablation.csv", index=False)
    print(summarize_filter_ablation(df_filter).to_string())
    print("\nBrand recovery fraction (has_method=0, has_model=1):")
    print(brand_recovery_fraction(df_filter).to_string())

    save_streamlit_summaries(df_primary, df_filter)

# df_primary= pd.read_csv("data/results/primary_ablation.csv")
# df_filter=pd.read_csv("data/results/filter_ablation.csv")
# save_streamlit_summaries(df_primary, df_filter)
