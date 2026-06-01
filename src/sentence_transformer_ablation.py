import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from src.model_building_utils import compute_mrr_grouped, compute_mrr
from src.ablation import load_data
import torch
import re
import logging

from pathlib import Path

# Specify the directory ('.' for current directory)
path = Path("./src")

# List only files
for file in path.iterdir():
    if file.is_file():
        print(file.name)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("logs/st_ablation.log"),
    ],
)
logger = logging.getLogger(__name__)


bertmodels = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "sentence-transformers/msmarco-distilbert-base-v4",
    "pritamdeka/S-PubMedBert-MS-MARCO",
    "neuml/pubmedbert-base-embeddings",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
]


elr, loinc, distractors, rn_stopwords = load_data()


def normalize_elr_for_st(text: str) -> str:
    """
    Lightweight normalization for sentence transformer input.
    Assumes clean_text has NOT been applied — input retains original
    punctuation, mixed case, and natural language surface forms.

    Standardizes the most variable surface forms to their most common
    biomedical pretraining representation without destroying clinical
    entity structure. Deliberately avoids:
      - clean_text token binding (FLUA, FLUB, SARSCOV2)
      - punctuation stripping (preserves RT-PCR, SARS-CoV-2)
      - uppercasing (dense models are case-aware)
    """
    if not text or not isinstance(text, str):
        return ""

    # Standardize COVID surface forms to canonical pretraining form
    text = re.sub(
        r"\bSARSCOV2\b|\bSARS[\s-]?COV[\s-]?2\b|\bSARS2\b"
        r"|\bCOVID[\s-]?19\b|\bCV[\s-]?19\b|\b2019[\s-]?NCOV\b"
        r"|\bSARS[\s-]CORONAVIRUS[\s-]2\b",
        "SARS-CoV-2",
        text,
        flags=re.IGNORECASE,
    )

    # Expand flu abbreviations to natural language
    # Handles F-A, F A, Flu A, FluA, FLUA → Influenza A
    text = re.sub(
        r"\b(?:F|FLU|INFLUENZA)[-\s]?(?:(?:VIRUS|TYPE)[-\s]?)?A\b",
        "Influenza A",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:F|FLU|INFLUENZA)[-\s]?(?:(?:VIRUS|TYPE)[-\s]?)?B\b",
        "Influenza B",
        text,
        flags=re.IGNORECASE,
    )

    # Expand method abbreviations to natural language forms
    text = re.sub(r"\bNAAT\b", "nucleic acid amplification", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNAA\b", "nucleic acid amplification", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\bRT[\s-]?PCR\b", "RT-PCR", text, flags=re.IGNORECASE
    )  # normalize spacing only
    # Normalize whitespace only — preserve punctuation and case
    return " ".join(text.split())


def compute_mrr_simple(candidates_df: pd.DataFrame, true_loinc: str) -> float:
    """Helper to calculate MRR from a ranked candidates dataframe."""
    matches = candidates_df[candidates_df["loinc_num"] == true_loinc]
    if not matches.empty:
        return 1.0 / matches.iloc[0]["rank"]
    return 0.0


def run_sentence_transformer_eval(
    df_loinc: pd.DataFrame,
    df_test: pd.DataFrame,
    model_link: str,
    top_k: int = 5,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Runs a fully vectorized zero-shot retrieval ablation using SapBERT.
    Assumes distractors are excluded.
    """

    df_test = df_test.copy().reset_index()
    print("Loading SapBERT...")
    model = SentenceTransformer(model_link)

    # Set model to evaluation mode for slight performance bump
    model.eval()

    # 1. Bulk Encode the Catalog (Anchor: Raw Long Common Name)
    print(f"Encoding {len(df_loinc)} LOINC codes...")
    catalog_embeddings = model.encode(
        df_loinc["long_common_name"].tolist(),
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # 2. Bulk Encode the Queries (Anchor: Normalized ELR Strings)
    print(f"Encoding {len(df_test)} test queries...")
    query_embeddings = model.encode(
        df_test["elr_name_normalized_st"].tolist(),
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # 3. Vectorized Matrix Similarity
    # Computes scores for ALL queries against ALL catalog entries simultaneously
    # Output shape: [len(df_test), len(df_loinc)]
    print("Computing similarity matrix...")
    cosine_scores = util.cos_sim(query_embeddings, catalog_embeddings)

    # 4. Extract Top-K across the entire matrix at once (dim=1 operates row-wise)
    print(f"Extracting Top-{top_k} rankings...")
    top_results = torch.topk(cosine_scores, k=min(top_k, len(df_loinc)), dim=1)

    # Move tensors back to CPU numpy arrays for pandas parsing
    indices_matrix = top_results.indices.cpu().numpy()
    scores_matrix = top_results.values.cpu().numpy()

    # 5. Assemble Results
    results = []

    # Since we vectorized the math, we only loop to build the final evaluation dataframe
    for idx, test_row in df_test.iterrows():
        true_loinc = test_row["loinc_num"]

        # Get the pre-calculated candidates for this specific row
        row_indices = indices_matrix[idx]
        row_scores = scores_matrix[idx]

        candidates = df_loinc.iloc[row_indices].copy()
        candidates["base_score"] = row_scores
        candidates["rank"] = range(1, len(candidates) + 1)

        # You can add corpus_text mapping here if you want it returned,
        # but SapBERT natively relies purely on long_common_name.

        # Compute Metrics
        mrr = compute_mrr(candidates, true_loinc)
        mrr_grouped = compute_mrr_grouped(candidates, test_row, true_loinc, df_loinc)

        # Grouped MRR calculation can be injected here using your existing function

        results.append(
            {
                "model_type": model_link,
                "elr_name": test_row["elr_name"],
                "elr_name_normalized_st": test_row["elr_name_normalized_st"],
                "elr_name_normalized": test_row["elr_name_normalized"],
                "true_loinc": true_loinc,
                "true_lcn": test_row["long_common_name"],
                "predicted_loinc": candidates.iloc[0]["loinc_num"],
                "predicted_lcn": candidates.iloc[0]["long_common_name"],
                "noise_level": test_row.get("noise_level"),
                "noise_compression": test_row.get("noise_compression"),
                "noise_omission": test_row.get("noise_omission"),
                "noise_corruption": test_row.get("noise_corruption"),
                "noise_total": test_row.get("noise_total"),
                "coverage_pattern": test_row.get("coverage_pattern"),
                "mrr": mrr,
                "mrr_grouped": mrr_grouped,
                "top1": int(candidates.iloc[0]["loinc_num"] == true_loinc),
                "top3": int(true_loinc in candidates.head(3)["loinc_num"].values),
                "top5": int(true_loinc in candidates.head(5)["loinc_num"].values),
                "has_method": test_row.get("has_method"),
                "has_model": test_row.get("has_model"),
                "has_specimen": test_row.get("has_specimen"),
                "specificity_mismatch": (mrr_grouped > mrr) and (mrr < 1.0),
            }
        )

    return pd.DataFrame(results)


def run_boosted_sentence_transformer(
    df_loinc: pd.DataFrame,
    df_test: pd.DataFrame,
    model_link: str,
    top_k: int = 5,
    batch_size: int = 64,
) -> pd.DataFrame:
    model = SentenceTransformer(model_link)

    df_test = df_test.copy().reset_index()
    # 1. Boost the Catalog (But keep ELR raw)
    # We use a more "Transformer-friendly" format
    df_loinc["boosted_text"] = df_loinc.apply(
        lambda x: f"{x['component']} method {x['method_typ']} system {x['system']}",
        axis=1,
    )

    # 2. Encode the Boosted Catalog
    catalog_embeddings = model.encode(
        df_loinc["boosted_text"].tolist(), convert_to_tensor=True
    )

    # 3. Encode the Raw ELR (No changes needed to ELR)
    query_embeddings = model.encode(
        df_test["elr_name_normalized"].tolist(), convert_to_tensor=True
    )

    # 4. Compute Similarity
    cosine_scores = util.cos_sim(query_embeddings, catalog_embeddings)

    # 4. Extract Top-K across the entire matrix at once (dim=1 operates row-wise)
    print(f"Extracting Top-{top_k} rankings...")
    top_results = torch.topk(cosine_scores, k=min(top_k, len(df_loinc)), dim=1)

    # Move tensors back to CPU numpy arrays for pandas parsing
    indices_matrix = top_results.indices.cpu().numpy()
    scores_matrix = top_results.values.cpu().numpy()

    # 5. Assemble Results
    results = []

    # Since we vectorized the math, we only loop to build the final evaluation dataframe
    for idx, test_row in df_test.iterrows():
        true_loinc = test_row["loinc_num"]

        # Get the pre-calculated candidates for this specific row
        row_indices = indices_matrix[idx]
        row_scores = scores_matrix[idx]

        candidates = df_loinc.iloc[row_indices].copy()
        candidates["base_score"] = row_scores
        candidates["rank"] = range(1, len(candidates) + 1)

        # You can add corpus_text mapping here if you want it returned,
        # but SapBERT natively relies purely on long_common_name.

        # Compute Metrics
        mrr = compute_mrr(candidates, true_loinc)
        mrr_grouped = compute_mrr_grouped(candidates, test_row, true_loinc, df_loinc)

        # Grouped MRR calculation can be injected here using your existing function

        results.append(
            {
                "model_type": model_link,
                "elr_name": test_row["elr_name"],
                "elr_name_normalized_st": test_row["elr_name_normalized_st"],
                "elr_name_normalized": test_row["elr_name_normalized"],
                "true_loinc": true_loinc,
                "true_lcn": test_row["long_common_name"],
                "predicted_loinc": candidates.iloc[0]["loinc_num"],
                "predicted_lcn": candidates.iloc[0]["long_common_name"],
                "noise_level": test_row.get("noise_level"),
                "noise_compression": test_row.get("noise_compression"),
                "noise_omission": test_row.get("noise_omission"),
                "noise_corruption": test_row.get("noise_corruption"),
                "noise_total": test_row.get("noise_total"),
                "coverage_pattern": test_row.get("coverage_pattern"),
                "mrr": mrr,
                "mrr_grouped": mrr_grouped,
                "top1": int(candidates.iloc[0]["loinc_num"] == true_loinc),
                "top3": int(true_loinc in candidates.head(3)["loinc_num"].values),
                "top5": int(true_loinc in candidates.head(5)["loinc_num"].values),
                "has_method": test_row.get("has_method"),
                "has_model": test_row.get("has_model"),
                "has_specimen": test_row.get("has_specimen"),
                "specificity_mismatch": (mrr_grouped > mrr) and (mrr < 1.0),
            }
        )

    return pd.DataFrame(results)


def run_sentence_transformer_ablation(elr, df_loinc) -> pd.DataFrame:
    """
    Secondary ablation: best corpus strategies x mixed model ngram/alpha sweep.
    No post-retrieval filtering.
    Returns a flat DataFrame of per-row results.
    """
    val_df = elr[elr["split"] == "val"].copy()
    all_results = []

    total = 2 * len(bertmodels)
    print(f"Secondary ablation: {total} runs")

    for i in range(len(bertmodels)):
        for strategy in ["regular_corpus", "boosted_corpus"]:
            print(f"Running {bertmodels[i]} with the {strategy}.")
            if strategy == "regular_corpus":
                results = run_sentence_transformer_eval(df_loinc, val_df, bertmodels[i])
            elif strategy == "boosted_corpus":
                results = run_boosted_sentence_transformer(
                    df_loinc, val_df, bertmodels[i]
                )
            results["model_num"] = i
            # results["bert_model"] = bertmodels[i]
            results["strategy"] = strategy
            all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


def summarize_ablation(df: pd.DataFrame):
    """Overall MRR by corpus x model x n_distractors."""
    return (
        df.groupby(["model_num", "model_type", "strategy"])
        .mrr_grouped.mean()
        .sort_values(ascending=False)
    )


def summarize_by_coverage(
    df: pd.DataFrame, mrr_col: str = "mrr_grouped"
) -> pd.DataFrame:
    """MRR by coverage_pattern for a results DataFrame."""
    return (
        df.groupby(["model_num", "model_type", "strategy", "coverage_pattern"])[mrr_col]
        .mean()
        .unstack("coverage_pattern")
    )

    # ------------------------------------------------------------------
    # SENTENCE TRANSFORMER
    # ------------------------------------------------------------------


def save_streamlit_summaries(
    df_st: pd.DataFrame,
    out_dir: str = "data/results/summary",
) -> None:
    """Generates all parquet summary files consumed by the Streamlit app's
    new load section. Call this at the bottom of ablation.py after the
    three ablation runs complete.

    Parameters
    ----------
    df_st: results from the st ablation
    out_dir    : destination folder (created if absent)

    Output files and their exact column sets
    -----------------------------------------
    st_by_config.parquet
        model_type, strategy, mrr_grouped, top1, top3, n

    st_by_coverage.parquet
        model_type, strategy, coverage_pattern, mrr_grouped, n

    st_by_noise.parquet
        model_type, strategy, noise_level, mrr_grouped, n

    st_by_method.parquet
        model_type, strategy, has_method, mrr_grouped, n
    """

    # Config-level (one row per model_type × strategy)
    import os

    os.makedirs(out_dir, exist_ok=True)

    (
        df_st.groupby(["model_type", "strategy"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/st_by_config.parquet", index=False)
    )

    # ST × coverage_pattern
    (
        df_st.groupby(["model_type", "strategy", "coverage_pattern"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/st_by_coverage.parquet", index=False)
    )

    # ST × noise_level
    (
        df_st.groupby(["model_type", "strategy", "noise_level"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/st_by_noise.parquet", index=False)
    )

    # ST × has_method
    (
        df_st.groupby(["model_type", "strategy", "has_method"])
        .agg(
            mrr_grouped=("mrr_grouped", "mean"),
            n=("mrr_grouped", "count"),
        )
        .reset_index()
        .to_parquet(f"{out_dir}/st_by_method.parquet", index=False)
    )

    # Report sizes
    import pathlib

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
    elr["elr_name_normalized_st"] = elr["elr_name"].map(normalize_elr_for_st)
    df_results = run_sentence_transformer_ablation(elr=elr, df_loinc=loinc)
    print(summarize_ablation(df_results))
    df_results.to_csv(
        "data/results/sentence_transformer_ablation_results.csv", index=False
    )
    print(summarize_by_coverage(df_results).loc[0].iloc[1].sort_values(ascending=False))
    save_streamlit_summaries(df_results)
