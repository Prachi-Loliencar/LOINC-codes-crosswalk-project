# LOINC Crosswalk Retrieval
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LOINC Crosswalk | Portfolio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    h1 { font-size: 1.6rem !important; }
    h3 { font-size: 1.05rem !important; color: #1e293b; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Display label dictionaries
# ---------------------------------------------------------------------------

CORPUS_LABELS = {
    "lcn_only": "LCN only",
    "combined": "2×LCN + RelNames + System",
    "lcn_filtered_rn_combined": "2×LCN + Filtered RelNames + System",
    "lcn_method_dict_combined": "LCN + Method Dict + System",
    "lcn_method_dict_filtered_rn": "2×LCN + Method Dict + Filtered RelNames + System",
    "component_weighted_method_dict": "2×Component + Method Dict + System",
}

MODEL_LABELS = {
    "tfidf_word_(1, 1)": "Word unigrams",
    "tfidf_word_(1, 2)": "Word uni+bigrams",
    "tfidf_char_(3, 5)": "Char n-grams (3–5)",
    "tfidf_mixed_None": "Mixed (word+char, α sweep)",
}

ST_MODEL_LABELS = {
    "pritamdeka/S-PubMedBert-MS-MARCO": "S-PubMedBert-MS-MARCO",
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6-v2",
    "sentence-transformers/msmarco-distilbert-base-v4": "msmarco-distilbert-v4",
    "neuml/pubmedbert-base-embeddings": "PubMedBERT-embeddings",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "SapBERT-fulltext",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token": "SapBERT-mean-token",
}

STRATEGY_LABELS = {
    "regular_corpus": "Regular (LCN only)",
    "boosted_corpus": "Boosted (LCN + component/method/system)",
}

# Hardcoded reference values for the best config:
# lcn_method_dict_combined, word (1,1), 0 distractors, no_filter condition.
# Source: filter_ablation.csv no_filter rows.
# Update these if you rerun the ablation with a different best config.
TFIDF_REF = {
    "mrr_grouped": 0.747,
    "top1": 0.282,
    "top3": 0.435,
    "noise": {"low": 0.762, "medium": 0.710, "high": 0.510},
    "has_method": {0: 0.697, 1: 0.763},
}

SUMMARY_DIR = "data/results/summary"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Primary ablation summary loaders
# Tabs that use these: sidebar, Tab 1, Tab 2, Tab 3
# ---------------------------------------------------------------------------


@st.cache_data
def load_primary():
    """
    Config-level summary: one row per (corpus_strategy, model_desc, n_distractors).
    Replaces the full primary_ablation.csv (~80 MB → ~5 KB).

    Used by:
      - Sidebar: populates corpus_strategy and model_desc filter lists
      - Tab 1: best_config identification, best_mrr, best_top1, best_top3,
               n_loinc_codes (true_loinc.nunique()), n_elr_strings
      - Tab 2: all bar charts, summary pivot table, lift chart
      - Tab 3: best_cs / best_md / best_nd selection for coverage breakdown
    """
    df = pd.read_parquet(f"{SUMMARY_DIR}/primary_by_config.parquet")
    df["corpus_label"] = (
        df["corpus_strategy"].map(CORPUS_LABELS).fillna(df["corpus_strategy"])
    )
    df["model_label"] = df["model_desc"].map(MODEL_LABELS).fillna(df["model_desc"])
    return df


@st.cache_data
def load_primary_coverage():
    """
    Coverage-pattern-level summary: one row per
    (corpus_strategy, model_desc, n_distractors, coverage_pattern).

    Used by:
      - Tab 3: per-pattern MRR bar chart, MRR/top1/top3/n table
    """
    df = pd.read_parquet(f"{SUMMARY_DIR}/primary_by_coverage.parquet")
    df["corpus_label"] = (
        df["corpus_strategy"].map(CORPUS_LABELS).fillna(df["corpus_strategy"])
    )
    df["model_label"] = df["model_desc"].map(MODEL_LABELS).fillna(df["model_desc"])
    return df


@st.cache_data
def load_primary_noise():
    """
    Noise-level summary: one row per
    (corpus_strategy, model_desc, n_distractors, noise_level).

    Used by:
      - Tab 3: coverage × noise heatmap (cross-joined with coverage in the tab)
    """
    df = pd.read_parquet(f"{SUMMARY_DIR}/primary_by_noise.parquet")
    df["corpus_label"] = (
        df["corpus_strategy"].map(CORPUS_LABELS).fillna(df["corpus_strategy"])
    )
    return df


@st.cache_data
def load_primary_coverage_noise():
    """
    Coverage pattern × Noise level summary.
    Used by Tab 3 noise heatmap.
    """
    return pd.read_parquet(f"{SUMMARY_DIR}/primary_by_coverage_noise.parquet")


# ---------------------------------------------------------------------------
# Filter ablation summary loaders
# Tabs that use these: Tab 4, Tab 6, Tab 7
# ---------------------------------------------------------------------------


@st.cache_data
def load_filter():
    """
    Config-level filter summary: one row per filter_applied condition.
    Replaces the full filter_ablation.csv (~60 MB → ~1 KB).

    Used by:
      - Tab 4: overall MRR/top1/top3/n by filter condition,
               oracle/brand lift metrics
      - Tab 6: TF-IDF no_filter baseline for ST comparison headline
      - Tab 7: val-side of val vs test headline table and bar chart
               (all three filter conditions)
    """
    df = pd.read_parquet(f"{SUMMARY_DIR}/filter_by_config.parquet")
    df["corpus_label"] = (
        df["corpus_strategy"].map(CORPUS_LABELS).fillna(df["corpus_strategy"])
    )
    return df


@st.cache_data
def load_filter_method():
    """
    Filter × has_method summary.

    Used by:
      - Tab 4: filter effect stratified by has_method bar chart
    """
    return pd.read_parquet(f"{SUMMARY_DIR}/filter_by_method.parquet")


@st.cache_data
def load_filter_coverage():
    """
    Filter × coverage_pattern summary.

    Used by:
      - Tab 4: filter effect by coverage pattern line/bar chart
      - Tab 6: TF-IDF no_filter coverage pattern baseline
      - Tab 7: val-side per-pattern table (no_filter condition only)
    """
    return pd.read_parquet(f"{SUMMARY_DIR}/filter_by_coverage.parquet")


@st.cache_data
def load_filter_noise():
    """
    Filter × noise dimension summaries (one parquet per noise column).
    Returns a dict keyed by noise column name.

    Used by:
      - Tab 7: val-side noise dimension breakdown (noise_level,
               noise_omission, noise_compression)
    """
    dims = ["noise_level", "noise_omission", "noise_compression"]
    return {
        dim: pd.read_parquet(f"{SUMMARY_DIR}/filter_by_{dim}.parquet") for dim in dims
    }


@st.cache_data
def load_filter_loinc():
    """
    Filter × LOINC row-level summary.
    Used by Tab 7 for per-LOINC scatter plot stability.
    """
    return pd.read_parquet(f"{SUMMARY_DIR}/filter_by_loinc.parquet")


# ---------------------------------------------------------------------------
# Sentence transformer summary loaders
# Tab that uses these: Tab 6
# ---------------------------------------------------------------------------


@st.cache_data
def load_st():
    """
    Config-level ST summary: one row per (model_type, strategy).
    Replaces sentence_transformer_ablation_results.csv (~200 MB → ~1 KB).

    Used by:
      - Tab 6: overall comparison table, boosted vs regular bar chart,
               MRR delta chart
    """
    df = pd.read_parquet(f"{SUMMARY_DIR}/st_by_config.parquet")
    df["model_label"] = df["model_type"].map(ST_MODEL_LABELS).fillna(df["model_type"])
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])
    return df


@st.cache_data
def load_st_coverage():
    """
    ST × coverage_pattern summary.

    Used by:
      - Tab 6: coverage pattern head-to-head chart and delta chart
    """
    df = pd.read_parquet(f"{SUMMARY_DIR}/st_by_coverage.parquet")
    df["model_label"] = df["model_type"].map(ST_MODEL_LABELS).fillna(df["model_type"])
    return df


@st.cache_data
def load_st_noise():
    """
    ST × noise_level summary.

    Used by:
      - Tab 6: noise robustness line chart and high-low delta table
    """
    df = pd.read_parquet(f"{SUMMARY_DIR}/st_by_noise.parquet")
    df["model_label"] = df["model_type"].map(ST_MODEL_LABELS).fillna(df["model_type"])
    return df


@st.cache_data
def load_st_method():
    """
    ST × has_method summary.

    Used by:
      - Tab 6: method token presence bar chart
    """
    df = pd.read_parquet(f"{SUMMARY_DIR}/st_by_method.parquet")
    df["model_label"] = df["model_type"].map(ST_MODEL_LABELS).fillna(df["model_type"])
    return df


# ---------------------------------------------------------------------------
# Full row-level loaders (kept as CSV — cannot be summarised)
# ---------------------------------------------------------------------------


@st.cache_data
def load_elr():
    """
    Full simulated ELR dataset. Cannot be summarised because Tab 5 uses
    every column: noise counts, component presence flags, coverage_pattern,
    analyte_len, elr_name_normalized, and noise × component cross-tabs.

    Used by:
      - Tab 1: live demo random sampler (val split rows)
      - Tab 5: all simulation quality sections (noise audit, structural
               composition, token frequency, component heatmaps)
    """
    return pd.read_csv("data/processed/elr_simulated.csv")


@st.cache_data
def load_loinc():
    """
    LOINC reference table (98 rows, ~100 KB). Kept as CSV because
    build_demo_index and compute_within_between need the full row-level
    table to fit TF-IDF and build the NN index.

    Used by:
      - Tab 1: build_demo_index (called via @st.cache_resource)
      - Tab 5: corpus geometry (within/between similarity, UMAP,
               token frequency corpus side)
    """
    from src.clinical_utils import clean_text
    from src.model_building_utils import (
        expand_loinc_lcn,
        compute_relatednames_stopwords,
    )

    loinc = pd.read_csv("data/processed/covid_surveillance_loinc.csv")
    loinc = loinc[~loinc.method_typ.isna()].copy()
    loinc["expanded_lcn"] = (
        loinc["long_common_name"].map(clean_text).map(expand_loinc_lcn)
    )
    return loinc


@st.cache_data
def load_rn_stopwords():
    """
    RelatedNames2 stopword set (tokens in >85% of rows).
    Derived from the LOINC table — small object, fast to compute.

    Used by:
      - Tab 5: build_corpus calls inside compute_within_between
               and the token frequency comparison
    """
    from src.model_building_utils import compute_relatednames_stopwords

    loinc = load_loinc()
    return compute_relatednames_stopwords(loinc, threshold=0.85)


@st.cache_data
def load_test_results():
    """
    Test-set filter ablation results. Small file (~1 MB) so CSV is fine.
    Returns (df, success_bool) to allow graceful degradation if the
    notebook that generates it hasn't been run yet.

    Used by:
      - Tab 1: test MRR delta on the overview metric card (non-blocking)
      - Tab 7: all sections — headline val vs test, per-coverage breakdown,
               noise robustness, per-LOINC scatter
    """
    path = "data/results/test_filter_ablation.csv"
    try:
        df = pd.read_csv(path)
        df["corpus_label"] = (
            df["corpus_strategy"].map(CORPUS_LABELS).fillna(df["corpus_strategy"])
        )
        return df, True
    except FileNotFoundError:
        return pd.DataFrame(), False


# ---------------------------------------------------------------------------
# Computed geometry cache (not a file load — kept here for co-location)
# ---------------------------------------------------------------------------


@st.cache_data
def compute_within_between(strategies: tuple, group_by: str = "method_typ"):
    """
    Pairwise cosine similarity within vs between method_typ groups,
    one row per corpus strategy. Accepts a tuple (not list) so it is
    hashable for st.cache_data — pass tuple(selected_geo_strategies).

    Used by:
      - Tab 5: Section D, within/between bar charts and Cohen's d chart
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    from src.model_building_utils import build_corpus

    loinc = load_loinc()
    rn_stopwords = load_rn_stopwords()
    df_unique = loinc.groupby("loinc_num").first().reset_index()

    records = []
    for strategy in strategies:
        corpus = build_corpus(df_unique, strategy, rn_stopwords)
        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
        )
        matrix = normalize(vec.fit_transform(corpus))
        sim = cosine_similarity(matrix)
        groups = df_unique[group_by].fillna("unknown").astype(str).values
        n = len(groups)
        within, between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                (within if groups[i] == groups[j] else between).append(sim[i, j])
        w, b = np.array(within), np.array(between)
        pooled_std = np.sqrt((w.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        cohens_d = float((w.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0
        ratio = float(w.mean() / b.mean()) if b.mean() > 0 else float("inf")
        records.append(
            {
                "strategy": strategy,
                "corpus_label": CORPUS_LABELS.get(strategy, strategy),
                "within": float(w.mean()),
                "between": float(b.mean()),
                "cohens_d": cohens_d,
                "ratio": ratio,
            }
        )
    return pd.DataFrame(records).sort_values("cohens_d", ascending=False)


# ---------------------------------------------------------------------------
# Eager startup load — only the files that every tab needs immediately.
# ELR, LOINC, and test results are loaded lazily inside their respective tabs.
# ---------------------------------------------------------------------------

try:
    df_primary = load_primary()
    df_primary_coverage = load_primary_coverage()
    df_primary_noise = load_primary_noise()
    df_filter = load_filter()
    df_filter_method = load_filter_method()
    df_filter_coverage = load_filter_coverage()
    df_filter_noise = load_filter_noise()  # dict keyed by noise column name
    df_st = load_st()
    df_st_coverage = load_st_coverage()
    df_st_noise = load_st_noise()
    df_st_method = load_st_method()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    missing_file = str(e)

# ELR and LOINC are loaded lazily inside Tab 5 to avoid blocking startup.
# Test results are loaded lazily inside Tab 1 (non-blocking) and Tab 7.

# End of load section

# ---------------------------------------------------------------------------
# Sidebar — global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## LOINC Crosswalk")
    st.markdown("**Electronic Lab Records  → LOINC Retrieval**")
    st.divider()

    if data_loaded:
        st.markdown(
            '<p class="section-header">Primary Ablation Filters</p>',
            unsafe_allow_html=True,
        )

        all_strategies = sorted(df_primary["corpus_strategy"].unique())
        selected_strategies = st.multiselect(
            "Corpus strategies",
            options=all_strategies,
            default=all_strategies,
            format_func=lambda x: CORPUS_LABELS.get(x, x),
            help="Filter which corpus strategies appear in Tab 2 charts",
        )

        all_models = sorted(df_primary["model_desc"].unique())
        selected_models = st.multiselect(
            "Vectorizer types",
            options=all_models,
            default=all_models,
            format_func=lambda x: MODEL_LABELS.get(x, x),
        )

        st.divider()
        st.markdown(
            '<p class="section-header">Coverage Pattern Filter</p>',
            unsafe_allow_html=True,
        )
        all_patterns = sorted(df_primary_coverage["coverage_pattern"].dropna().unique())
        selected_patterns = st.multiselect(
            "Coverage patterns (Tab 3)",
            options=all_patterns,
            default=all_patterns,
            help="A=analyte, M=method, S=specimen signal present in ELR string",
        )

        st.divider()
        st.markdown(
            "**Metric:** Specimen-aware grouped MRR  \n"
            "Equivalence groups account for LOINC's catchall vs specific specimen system ambiguity.",
            help="mrr_grouped treats clinically equivalent LOINC codes as correct retrievals",
        )

        st.divider()
        st.markdown(
            '<p class="section-header">Tab 5 — Corpus Geometry</p>',
            unsafe_allow_html=True,
        )
        all_corpus_strategies_for_geo = [
            "lcn_only",
            "lcn_method_dict_combined",
            "lcn_method_dict_filtered_rn",
            "component_weighted_method_dict",
        ]
        selected_geo_strategies = st.multiselect(
            "Strategies for within/between analysis",
            options=all_corpus_strategies_for_geo,
            default=all_corpus_strategies_for_geo,
            format_func=lambda x: CORPUS_LABELS.get(x, x),
            help="Recomputes on change — may take a few seconds",
        )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("LOINC Crosswalk: Electronic Lab Record to LOINC Code Retrieval Benchmark")
st.markdown(
    "TF-IDF retrieval system mapping noisy COVID-19 Electronic Lab Report test name strings  "
    "to standardized LOINC codes. LOINC (Logical Observation Identifiers Names and Codes) is a universal standard for identifying medical laboratory tests - a shared vocabulary that allows different hospitals, labs, and health systems to refer to the same test using the same code, regardless of what the local instrument or lab vendor calls it. "
    "Evaluated via **specimen-aware grouped MRR** across "
    "a structured ablation of corpus design, vectorizer type, and post-retrieval filtering."
)

if not data_loaded:
    st.error(
        f"Could not load results files. Run the ablation scripts first.\n\n`{missing_file}`"
    )
    st.info(
        "Expected files:\n"
        "- `data/results/primary_ablation.csv`\n"
        "- `data/results/secondary_ablation.csv`\n"
        "- `data/results/filter_ablation.csv`"
    )
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Overview",
        "Primary Ablation",
        "Coverage Analysis",
        "Filter Ablation",
        "Simulation & Corpus",
        "TF-IDF vs Sentence Transformers",
        "Test Set Results",
    ]
)

# ===========================================================================
# TAB 1 — Overview
# ===========================================================================
with tab1:
    st.markdown("### What This Project Does")
    st.markdown("""
Real-world Electronic Lab Reports (ELRs) contain free-text or semi-structured test name strings
that vary widely across laboratories, including, abbreviations, typos, missing fields and instrument-specific
naming conventions. Mapping these to standardized **LOINC codes** is a recurring bottleneck
in public health informatics.

This project builds a **TF-IDF retrieval system** that maps simulated noisy COVID-19 related ELR strings
to the correct LOINC code, evaluated against 98 COVID-19 SARS-CoV-2 LOINC codes. A sentence transformer comparison establishes whether dense retrieval adds value over the sparse TF-IDF baseline for this terminological task.        """)

    st.markdown("### Project at a Glance")

    # Key result metrics — computed from primary ablation data
    best_row = (
        df_primary.groupby(["corpus_strategy", "model_desc", "n_distractors"])[
            "mrr_grouped"
        ]
        .mean()
        .reset_index()
    )
    best_config = best_row.loc[best_row["mrr_grouped"].idxmax()]
    best_mrr = best_config["mrr_grouped"]

    # Exact-match top-1 for best config
    best_mask = (
        (df_primary["corpus_strategy"] == best_config["corpus_strategy"])
        & (df_primary["model_desc"] == best_config["model_desc"])
        & (df_primary["n_distractors"] == best_config["n_distractors"])
    )
    best_df = df_primary[best_mask]
    best_top1 = best_df["top1"].mean()
    best_top3 = best_df["top3"].mean()
    n_loinc_codes = df_primary["n_loinc_codes"].iloc[0]
    n_elr_strings = int(best_df["n"].iloc[0])

    # Load test results for the delta display (non-blocking)
    df_test_ov, test_loaded_ov = load_test_results()
    test_nf_mrr = None
    if test_loaded_ov and not df_test_ov.empty:
        _test_nf = df_test_ov[df_test_ov["filter_condition"] == "no_filter"]
        if not _test_nf.empty:
            test_nf_mrr = _test_nf["mrr_grouped"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_str = f"{test_nf_mrr:.3f} on test" if test_nf_mrr is not None else None
        st.metric(
            "Best Grouped MRR (Val)",
            f"{best_mrr:.3f}",
            delta=delta_str,
            delta_color="off",
            help="Validation set MRR (best ablation config). Delta shows held-out test set result.",
        )
    with col2:
        st.metric(
            "Top-1 Accuracy",
            f"{best_top1:.1%}",
            help="Exact LOINC code match at rank 1 (validation set)",
        )
    with col3:
        st.metric("Top-3 Accuracy", f"{best_top3:.1%}")
    with col4:
        st.metric(
            "LOINC Codes Evaluated",
            str(n_loinc_codes),
            help="COVID-19 SARS-CoV-2 codes only",
        )

    col5, col6 = st.columns(2)
    with col5:
        st.metric(
            "Best Corpus Strategy",
            CORPUS_LABELS.get(
                best_config["corpus_strategy"], best_config["corpus_strategy"]
            ),
            help="Long common name with method tokens expanded via a dictionary",
        )
    with col6:
        st.metric(
            "Vectorizer",
            MODEL_LABELS.get(best_config["model_desc"], best_config["model_desc"]),
            help=f"{int(best_config['n_distractors'])} non-COVID distractor codes added to corpus",
        )
    st.divider()

    st.markdown("### Pipeline")
    st.markdown("""
1. **Preprocessing:** Merge CDC LIVD device submissions with the LOINC table;
   explode multi-specimen rows; deduplicate on a clinical key; filter to ≥3 seeds per LOINC code
2. **ELR Simulation:** Generate realistic noisy lab strings from LIVD vendor analyte names
   with controlled perturbation across three noise dimensions: corruption (character-level typos),
   compression (alternate surface forms of the same semantic entity), and omission (signal deleted entirely)
3. **Corpus Construction:** 5 corpus strategies combining LOINC long common names, method token
   dictionaries, system expansions, and filtered RelatedNames2
4. **TF-IDF Retrieval:** Word, character, and mixed vectorizers (α controls word vs char contribution);
   cosine similarity via nearest-neighbour index
5. **Evaluation:** Specimen-aware grouped MRR; equivalence grouping absorbs LOINC's
   catchall vs specific specimen ambiguity
6. **Post-Retrieval Filtering:** Oracle filter (ground-truth metadata upper bound) and
   brand-imputation filter (production-feasible) evaluated on the best config
7. **Sentence Transformer Retrieval:** 6 models × 2 corpus strategies; natural language
   corpus without TF-IDF tokenization to preserve semantic structure
        """)

    st.divider()
    st.markdown("### Key Findings")
    st.markdown("""
- **The vocabulary gap is the central challenge.** Only 7.8% of ELR query tokens appear
  in the raw LOINC table. Real-world lab senders use abbreviations, brand names, and informal
  terminology that LOINC's formal vocabulary doesn't anticipate. Every major design decision
  in this project, including  corpus strategies, expansion dictionaries, LCN repetition and distractor
  IDF correction, is a response to this gap. Closing it is harder than it looks - relatednames2
  adds synonym coverage but dilutes discriminative signal; the method dictionary is the solution
  that bridges the gap without introducing noise.

- **The vocabulary bridge works.** After expansion, TF-IDF achieves **0.737 grouped MRR on
  analyte-only strings** (`A` pattern: no method, no specimen token present). Retrieval is
  succeeding purely on analyte vocabulary overlap, validating that the expansion dictionaries
  are doing meaningful work rather than adding noise. Sentence transformer achieves only 0.524
  on the same strings despite having richer semantic representations.

- **Corpus strategy is the dominant design lever.** Switching from `lcn_only` (MRR 0.552)
  to `lcn_method_dict_combined` (MRR 0.747) gains **0.195 grouped MRR** at 0 distractors.
  The best vectorizer choice within the same strategy spans 0.165 MRR (word unigrams 0.747
  vs char n-grams 0.583), confirming that corpus design matters as much as model choice.
  Explicit method dictionary expansion outperforms relatednames2 augmentation even with LCN
  repetition to counteract dilution.

- **Distractor codes correct IDF.** COVID-specific tokens (`SARSCOV2`, `NAAT`) get
  near-zero IDF in a COVID-only corpus; non-COVID respiratory distractors restore
  discriminative weight. The effect is strategy-dependent: `lcn_method_dict_combined` degrades
  monotonically with distractors while `component_weighted_method_dict` benefits for the best performing model,
  suggesting the distractor benefit may not generalise beyond this specific corpus.

- **Oracle lift is concentrated on specimen-containing patterns.** Perfect metadata
  extraction would gain 0.064 MRR on `M+S` and 0.038 on `A+M+S`, but only 0.008 on `A`
  and nothing on `I`. Specimen signal is present in the ELR string for these patterns but
  the corpus does not fully leverage it without filtering. The brand filter contributes
  effectively zero lift across all patterns — it fires only when a model token is present,
  which is rare in the val set — confirming the negligible overall gain of 0.001 MRR
  (0.747 → 0.748).

- **ST wins only on method-absent, interpretation token strings.** Sentence transformer
  outperforms TF-IDF on `I` (0.128 MRR advantage) and `M+I` (0.089 MRR advantage) patterns,
  where no discriminative token signal is present for sparse retrieval to match. TF-IDF leads
  on all other 14 coverage patterns, with the largest advantages on `S` (0.248 MRR) and
  `A` (0.213 MRR).

- **Catchall vs specific ambiguity is a corpus artifact, not a retrieval failure.** 33.6% of wrong top-1 predictions are catchall-to-specific system mismatches. Of these,
  53% are absorbed by specimen-aware grouped MRR as clinically equivalent retrievals,
  confirming the issue is a LOINC device registry artifact. The remaining genuine retrieval
  failures (14.8% of the val set with MRR=0) are concentrated in low-information coverage
  patterns (`I`, `NONE`, `M+I`).
    """)

    st.divider()
    st.markdown("### Live Retrieval Demo")
    st.markdown(
        "Sample a random ELR string from the validation set and retrieve the top-5 "
        "LOINC candidates using the best TF-IDF configuration from the ablation "
        "(`lcn_method_dict_combined` · word unigrams). "
        "Click repeatedly to see different difficulty levels and coverage patterns."
    )
    st.caption(
        "⚠️ This system covers **COVID-19 SARS-CoV-2 surveillance codes only** (36 LOINC codes). "
        "Custom strings unrelated to COVID-19 will still return a ranked result as "
        "the nearest-neighbor index always retrieves, but the output will not be meaningful."
    )

    # ------------------------------------------------------------------
    # Index — built once and cached for the session lifetime
    # ------------------------------------------------------------------
    @st.cache_resource
    def build_demo_index():
        """
        Builds the TF-IDF retrieval index using the best ablation config:
          corpus_strategy : lcn_method_dict_combined
          model_type      : tfidf_word (1,1)
        Returns (vectorizer, nn_index, df_loinc_corpus).
        """
        from src.clinical_utils import clean_text as _ct
        from src.model_building_utils import (
            expand_loinc_lcn,
            build_corpus,
            build_tfidf_index,
            build_nn_index,
            compute_relatednames_stopwords,
        )

        loinc = pd.read_csv("data/processed/covid_surveillance_loinc.csv")
        loinc = loinc[~loinc.method_typ.isna()].copy()
        loinc["expanded_lcn"] = loinc["long_common_name"].map(_ct).map(expand_loinc_lcn)
        rn_stopwords = compute_relatednames_stopwords(loinc, threshold=0.85)
        corpus = build_corpus(loinc, "lcn_method_dict_combined", rn_stopwords)
        loinc["corpus_text"] = corpus
        vectorizer, corpus_matrix = build_tfidf_index(corpus, "tfidf_word", (1, 1))
        nn = build_nn_index(corpus_matrix, n_neighbors=5)
        return vectorizer, nn, loinc

    # ------------------------------------------------------------------
    # Input controls
    # ------------------------------------------------------------------
    input_mode = st.radio(
        "Input mode",
        ["Random from validation set", "Custom string"],
        horizontal=True,
        key="demo_input_mode",
    )

    if input_mode == "Random from validation set":
        if st.button("🎲 Sample random ELR string", key="demo_random"):
            try:
                df_val = load_elr()
                # Filter to val split if the split column exists
                if "split" in df_val.columns:
                    df_val = df_val[df_val["split"] == "val"]
                sample = df_val.sample(1, random_state=None).iloc[0]
                st.session_state["demo_elr_input"] = sample["elr_name"]
                st.session_state["demo_elr_meta"] = {
                    "true_loinc": sample.get("loinc_num", "—"),
                    "coverage_pattern": sample.get("coverage_pattern", "—"),
                    "noise_level": sample.get("noise_level", "—"),
                    "noise_total": int(sample.get("noise_total", 0)),
                    "has_method": bool(sample.get("has_method", False)),
                    "has_specimen": bool(sample.get("has_specimen", False)),
                    "specimen_norm": sample.get("specimen_norm", "UNKNOWN"),
                }
            except FileNotFoundError:
                st.warning(
                    "`data/processed/elr_simulated.csv` not found. "
                    "Run `elr_simulation_main.py` first."
                )

        # Show the sampled string if one is in session state
        if "demo_elr_input" in st.session_state:
            st.code(st.session_state["demo_elr_input"], language="text")
            meta = st.session_state.get("demo_elr_meta", {})
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("True LOINC", meta.get("true_loinc", "—"))
            m2.metric("Coverage Pattern", meta.get("coverage_pattern", "—"))
            m3.metric("Noise Level", str(meta.get("noise_level", "—")).title())
            m4.metric("Has Method", "Yes" if meta.get("has_method") else "No")
            m5.metric("Has Specimen", "Yes" if meta.get("has_specimen") else "No")
            elr_to_retrieve = st.session_state["demo_elr_input"]
        else:
            elr_to_retrieve = None
            st.info("Click **Sample random ELR string** to begin.")

    else:  # Custom string
        custom_input = st.text_input(
            "Enter a COVID-19 ELR string:",
            placeholder="e.g. COVID-19 PCR NASOPHARYNGEAL SWAB",
            key="demo_custom_input",
        )
        elr_to_retrieve = custom_input.strip() if custom_input.strip() else None
        if elr_to_retrieve:
            st.caption(
                "Custom strings are preprocessed with `clean_text` + `normalize_elr` "
                "before retrieval — same pipeline as the ablation evaluation."
            )

    # ------------------------------------------------------------------
    # Retrieval — fires when input is available and button clicked
    # ------------------------------------------------------------------
    run_retrieval = st.button(
        "▶ Retrieve",
        key="demo_retrieve",
        type="primary",
        disabled=(elr_to_retrieve is None),
    )

    if run_retrieval and elr_to_retrieve:
        try:
            from src.clinical_utils import clean_text as _ct2
            from src.model_building_utils import normalize_elr as _ne, retrieve

            normalized = _ne(_ct2(elr_to_retrieve))

            with st.spinner("Retrieving…"):
                vec, nn, df_corpus = build_demo_index()
                candidates = retrieve(normalized, vec, df_corpus, nn)

            top_score = candidates.iloc[0]["base_score"]
            score_gap = (
                top_score - candidates.iloc[1]["base_score"]
                if len(candidates) > 1
                else 0.0
            )
            # Approximate val-set quantile thresholds
            q33, q67 = 0.35, 0.55
            tier = (
                "🟢 HIGH"
                if top_score >= q67
                else "🟡 MEDIUM"
                if top_score >= q33
                else "🔴 LOW — manual review recommended"
            )

            st.markdown("---")
            col_norm, col_conf = st.columns(2)
            with col_norm:
                st.markdown("**Normalized query (retrieval input):**")
                st.code(normalized, language="text")
            with col_conf:
                st.markdown("**Retrieval confidence:**")
                st.markdown(f"**{tier}**")
                st.caption(
                    f"Top cosine score: `{top_score:.3f}` · Score gap: `{score_gap:.3f}`  \n"
                    "Confidence tiers use val-set quantile thresholds — "
                    "recalibrate on real ELR data for production use."
                )

            st.markdown("#### Top-5 LOINC Candidates")

            # If we have the true LOINC from a val set sample, annotate it
            true_loinc = st.session_state.get("demo_elr_meta", {}).get("true_loinc")
            if input_mode == "Custom string":
                true_loinc = None

            # Compute the grouped equivalence set for the true LOINC so we can
            # correctly label clinically valid retrievals that are not exact matches.
            # This mirrors the grouped MRR logic used in the ablation evaluation.
            valid_loincs = set()
            if true_loinc and true_loinc != "—":
                try:
                    from src.model_building_utils import get_valid_loincs

                    elr_meta = st.session_state.get("demo_elr_meta", {})
                    elr_row_for_grouping = pd.Series(
                        {
                            "specimen_norm": elr_meta.get("specimen_norm", "UNKNOWN"),
                        }
                    )
                    valid_loincs = get_valid_loincs(
                        elr_row_for_grouping, df_corpus, true_loinc
                    )
                except Exception:
                    # Fallback: exact match only if grouping fails
                    valid_loincs = {true_loinc}

            for _, row in candidates.iterrows():
                is_exact = (true_loinc is not None) and (row["loinc_num"] == true_loinc)
                is_equivalent = (
                    (true_loinc is not None)
                    and (row["loinc_num"] in valid_loincs)
                    and not is_exact
                )
                rank_icon = "🥇" if row["rank"] == 1 else f"#{int(row['rank'])}"

                if is_exact:
                    result_tag = "  ✅ exact match"
                elif is_equivalent:
                    result_tag = "  🟦 clinically equivalent"
                else:
                    result_tag = ""

                with st.expander(
                    f"{rank_icon}  `{row['loinc_num']}`  —  "
                    f"score: {row['base_score']:.3f}{result_tag}",
                    expanded=(row["rank"] == 1),
                ):
                    st.markdown(f"**{row['long_common_name']}**")
                    if is_equivalent:
                        st.caption(
                            "🟦 Clinically equivalent to ground truth — same component "
                            "and method, compatible specimen system. Counted as correct "
                            "under specimen-aware grouped MRR."
                        )
                    elif is_exact:
                        st.caption("✅ Exact match with ground truth LOINC code.")
                    st.caption(
                        f"Corpus text (truncated): `{str(row['corpus_text'])[:130]}…`"
                    )

            # Ground truth outcome summary for val set samples
            if true_loinc and true_loinc != "—":
                predicted = candidates.iloc[0]["loinc_num"]
                predicted_is_valid = (
                    predicted in valid_loincs or predicted == true_loinc
                )
                true_in_top5 = true_loinc in candidates["loinc_num"].values
                valid_in_top5 = any(
                    c in valid_loincs or c == true_loinc
                    for c in candidates["loinc_num"].values
                )

                if predicted == true_loinc:
                    st.success(
                        f"✅ Top-1 exact match — ground truth `{true_loinc}` ranked first."
                    )
                elif predicted_is_valid:
                    st.success(
                        f"🟦 Top-1 clinically equivalent — `{predicted}` shares the same "
                        f"component and method as ground truth `{true_loinc}` with a "
                        f"compatible specimen system. Counted as correct under grouped MRR."
                    )
                elif valid_in_top5:
                    # Find the best-ranked valid code
                    best_valid_rank = min(
                        int(
                            candidates.loc[candidates["loinc_num"] == c, "rank"].values[
                                0
                            ]
                        )
                        for c in candidates["loinc_num"].values
                        if c in valid_loincs or c == true_loinc
                    )
                    st.warning(
                        f"⚠️ A clinically valid code is ranked #{best_valid_rank} — "
                        f"not top-1 but within top-5. "
                        f"Ground truth: `{true_loinc}`."
                    )
                else:
                    true_lcn = df_corpus[df_corpus["loinc_num"] == true_loinc][
                        "long_common_name"
                    ].values
                    true_lcn_str = true_lcn[0] if len(true_lcn) > 0 else "—"
                    n_valid = len(valid_loincs)
                    st.error(
                        f"❌ No clinically valid code in top-5.  \n"
                        f"**Ground truth:** `{true_loinc}` — {true_lcn_str}  \n"
                        f"**Valid equivalence set size:** {n_valid} code(s)  \n"
                        "Check coverage pattern and noise level above for context."
                    )

        except FileNotFoundError as e:
            st.error(
                f"Required data file not found: `{e}`.  \n"
                "Run preprocessing and simulation scripts before launching the app."
            )
        except Exception as e:
            st.error(f"Retrieval error: `{e}`")

    st.divider()
    st.markdown("### Evaluation metric")
    st.markdown(
        """The evaluation metric primarily used for the ablation is the **Mean Reciprocal Rank (MRR)**:"""
    )
    st.latex(r"\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}")
    st.markdown(
        "Since LOINC axes do not uniquely identify codes, particularly the **specimen system** axis, "
        "where manufacturers inconsistently chose between specific codes (e.g., `Nph` for nasopharynx) "
        "and catchall codes (e.g., `Respiratory System Specimen`) for clinically identical tests,"
        "exact-match MRR would penalize correct predictions. "
        "\n\n"
        "**Grouped MRR** instead defines an equivalence set per ELR query that includes all clinically valid LOINC codes "
        "(same component and method, compatible specimen system). A prediction is correct if it lands in this set. "
        "This separates genuine retrieval failures from device registry artifacts and is the appropriate metric "
        "for evaluating health informatics systems where clinical correctness doesn't necessarily coincide with database exactness."
    )

    #    col_left, col_right = st.columns([1.1, 0.9])

    # with col_left:

    #    with col_right:

    # st.markdown("### Best Configuration")
    # st.info(
    #     f"**Corpus:** {CORPUS_LABELS.get(best_config['corpus_strategy'], best_config['corpus_strategy'])}  \n"
    #     f"**Vectorizer:** {MODEL_LABELS.get(best_config['model_desc'], best_config['model_desc'])}  \n"
    #     f"**Distractors:** {int(best_config['n_distractors'])} non-COVID respiratory codes  \n"
    #     f"**Grouped MRR:** {best_mrr:.3f}"
    # )

    st.markdown("### Variable Description")
    st.markdown("""
        - **long_common_name (LCN):** Natural language text in the LOINC including a description of all the axes of the test              
        - **system/specimen/specimen_norm:** Describes the location the specimen is taken from (eg. "Nasopharynx" )
        - **method/method_typ:** Describes the type of test, broadly divided into two classes - NAAT and antigen
        - **model:** The specific model of the test kit or instrument (manufacturer specific)
        - **analyte/component:** Describes the component of the specimen that is being tested (eg. RNA)
        - **relatednames2:** Provides a list of synonyms, keywords, abbreviations, and related terms associated with a LOINC code (including method)
        - **method_class:** Class describing the method - can be either NAAT or Antigen.""")

    st.markdown("### Data Sources")
    st.markdown("""
    | Source | Description |
    |--------|-------------|
    | CDC LIVD Table | FDA-authorized device submissions mapping kits to LOINC |
    | LOINC Table | Component, system, method, long common name |
    | Simulated ELR | Generated from LIVD vendor analyte names + perturbations |
            """)


# ===========================================================================
# TAB 2 — Primary Ablation
# ===========================================================================
with tab2:
    st.markdown("### Primary Ablation: Corpus Strategy × Vectorizer × Distractor Count")
    st.markdown("""
        - **Corpus Strategy** tests the use of different columns of the LOINC table and method of expanding tokens to match ELR language ([see descriptions of strategies](#strategy-definitions)). 
        - **Vectorizer** choices for the TF-IDF vectorizer included the choice of Word-based, Char-based and mixed models as well as the choice of n-grams.   
        - **Distractor Count** indicates the addition of distractor codes to the LOINC table. The vanilla retrieval with 0 distractors is done against covid surveillance codes (all codes in the CDC LIVD table). To assess the impact of distractor codes on the performance of TF-IDF, 143 respiratory disease codes that present similarly to COVID-19 were considered from the raw LOINC table; each row was required to have **Method** and **System** values that were already present in the covid surveillance codes to create realistic distractors.""")

    st.markdown(
        "Each bar below is the mean **grouped MRR** over the validation set (5280 ELR strings). "
        "Use the sidebar to filter strategies and vectorizer types."
    )

    # Apply sidebar filters
    df2 = df_primary[
        df_primary["corpus_strategy"].isin(selected_strategies)
        & df_primary["model_desc"].isin(selected_models)
    ].copy()

    if df2.empty:
        st.warning("No data matches the current sidebar filters.")
        st.stop()

    summary = df2.copy()
    summary["distractor_label"] = summary["n_distractors"].apply(
        lambda x: f"{int(x)} distractors"
    )

    # Sort by MRR at 0 distractors descending for consistent ordering
    order = (
        summary[summary["n_distractors"] == 0]
        .sort_values("mrr_grouped", ascending=False)["corpus_label"]
        .tolist()
    )

    # col_chart, col_table = st.columns([1.6, 1])

    # with col_chart:
    # Grouped bar: one bar per (corpus × distractor count), faceted by model
    fig = px.bar(
        summary,
        x="mrr_grouped",
        y="corpus_label",
        color="distractor_label",
        facet_col="model_label",
        facet_col_wrap=2,
        orientation="h",
        barmode="group",
        category_orders={"corpus_label": order},
        # color_discrete_map={
        #     "0 distractors": "#94a3b8",
        #     "100 distractors": "#103879",
        # },
        labels={"mrr_grouped": "Grouped MRR", "corpus_label": "Corpus Strategy"},
        height=420,
    )
    fig.update_layout(
        margin=dict(l=0, r=10, t=40, b=10),
        legend_title_text="Distractor Count",
        font_size=11,
    )
    fig.update_xaxes(range=[0, 1], dtick=0.2)
    # Add a vertical reference line at random baseline (~0.15 for this corpus size)
    for ax in fig.layout:
        if ax.startswith("xaxis"):
            fig.add_vline(
                x=0.15,
                line_dash="dot",
                line_color="#434141",
                annotation_text="Random",
                annotation_position="top right",
                row="all",
                col="all",
            )
    st.plotly_chart(fig, use_container_width=True)

    #    with col_table:
    st.markdown("**Summary table** — mean grouped MRR")
    # pivot = (
    #     summary[summary["n_distractors"] == 143]
    #     .pivot_table(index="corpus_label", columns="model_label", values="mrr_grouped")
    #     .round(3)
    # )
    # 1. Pivot without resetting the index
    # This keeps corpus_label as the first level and n_distractors as the second
    pivot_multi = summary.pivot_table(
        index=["corpus_label", "n_distractors"],
        columns="model_label",
        values="mrr_grouped",
    ).round(3)

    # 2. Display with the index intact
    st.dataframe(
        pivot_multi.style.highlight_max(axis=0, color="#bbf7d0"),
        use_container_width=True,
        # Note: Do NOT use hide_index=True here, otherwise the merged cells disappear!
    )
    st.caption("Green = best per vectorizer column.")

    st.divider()
    st.markdown("### MRR Lift from Distractors")
    st.markdown(
        "The delta between 0 and 143 distractors shows how much IDF correction "
        "helps each strategy. Strategies that already include a method token dictionary "
        "benefit most because those tokens are also the ones with depressed IDF in a COVID-only corpus."
    )

    lift = summary.pivot_table(
        index=["corpus_label", "model_label"],
        columns="n_distractors",
        values="mrr_grouped",
    ).reset_index()
    lift.columns.name = None
    if 0 in lift.columns and 143 in lift.columns:
        lift["lift"] = lift[143] - lift[0]
        lift_sorted = lift.sort_values("lift", ascending=False)

        fig_lift = px.bar(
            lift_sorted,
            x="lift",
            y="corpus_label",
            color="model_label",
            orientation="h",
            barmode="group",
            labels={
                "lift": "MRR lift (143 distractors − 0)",
                "corpus_label": "Corpus Strategy",
            },
            height=320,
            # color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_lift.add_vline(x=0, line_color="#64748b", line_width=1)
        fig_lift.update_layout(
            margin=dict(l=0, r=10, t=20, b=10), legend_title_text="Vectorizer"
        )
        st.plotly_chart(fig_lift, use_container_width=True)

    corpus_descriptions = {
        "LCN only": (
            "Long common name only (baseline). No system or method expansion."
        ),
        "2×LCN + RelNames + System": (
            "Long common name (repeated 2×) + system expansion + relatednames2. "
            "LCN is repeated to counteract TF dilution from the long relatednames2 field as "
            "repetition raises TF counts for LCN tokens without affecting IDF."
        ),
        "2×LCN + Filtered RelNames + System": (
            "Long common name (repeated 2×) + system expansion + relatednames2 filtered to "
            "remove tokens appearing in >85% of codes (uninformative stopwords). "
            "Reduces dilution while retaining some relatednames2 signal."
        ),
        "LCN + Method Dict + System": (
            "Long common name + system expansion + method dictionary expansion. "
            "Best-performing strategy. Replaces relatednames2 entirely with a compact "
            "domain dictionary (e.g. Probe.amp.tar → 'NAAT NAA PCR RT-PCR QPCR'). "
            "No LCN repetition needed — the dilution problem is eliminated at its source."
        ),
        "2×LCN + Method Dict + Filtered RelNames + System": (
            "Long common name (repeated 2×) + system expansion + method dictionary + "
            "filtered relatednames2. Combines method dictionary and filtered synonyms, "
            "but performs below LCN + Method Dict + System, suggesting residual "
            "relatednames2 content adds noise even after filtering."
        ),
        "2×Component + Method Dict + System": (
            "LOINC component field (repeated 2×) + method dictionary expansion + system expansion. "
            "Upweights the component axis on the assumption it is more discriminative than "
            "the full long common name. Performs well with distractors but shows unstable "
            "generalization behavior as distractor count increases."
        ),
    }

    desc_df = pd.DataFrame(
        list(corpus_descriptions.items()), columns=["Strategy", "Description"]
    )

    st.markdown("### Strategy Definitions")
    st.markdown(
        "All strategies except `lcn_only` include system expansion "
        "(e.g. `Nph` → 'Nasopharynx Nasopharyngeal NP NPH') and LCN expansion "
        "(e.g. 'NAA with probe detection' → appends 'NAAT NAA PCR'). "
        "Strategies that include relatednames2 repeat the LCN to counteract "
        "signal dilution from that field's length and heterogeneity. "
        "Other variants (eg. `axes_only`) were tested during development but "
        "excluded from the reported ablation - see the GitHub README for details."
    )
    st.table(desc_df)


# ===========================================================================
# TAB 3 — Coverage Pattern Analysis
# ===========================================================================
with tab3:
    st.markdown("### Retrieval Performance by ELR Information Content")
    st.markdown(
        "Coverage pattern encodes which signal types are present in the ELR string: "
    )
    st.markdown(""" 
        - **A** = analyte token
        - **M** = method token
        - **S** = specimen token
        - **I** = non-discriminative noise tokens like "interpretation", "result" etc. """)
    st.markdown("This tab shows how grouped MRR degrades as information is removed.")

    # Use best config from primary ablation for this analysis
    best_cs = best_config["corpus_strategy"]
    best_md = best_config["model_desc"]
    best_nd = best_config["n_distractors"]

    df3 = df_primary_coverage[
        (df_primary_coverage["corpus_strategy"] == best_cs)
        & (df_primary_coverage["model_desc"] == best_md)
        & (df_primary_coverage["n_distractors"] == best_nd)
        & (df_primary_coverage["coverage_pattern"].isin(selected_patterns))
    ].copy()

    col_sel, _ = st.columns([2, 2])
    with col_sel:
        st.info(
            f"Showing best config: **{CORPUS_LABELS.get(best_cs, best_cs)}** "
            f"/ **{MODEL_LABELS.get(best_md, best_md)}** "
            f"/ {int(best_nd)} distractors"
        )

    if df3.empty:
        st.warning("No data for the selected coverage patterns.")
    else:
        # MRR by coverage pattern
        cov_summary = df3.copy()

        # Canonical sort order by information content
        COVERAGE_ORDER = [
            "A+M+S+I",
            "A+M+S",
            "A+M+I",
            "A+M",
            "A+S+I",
            "A+S",
            "A+I",
            "A",
            "M+S",
            "M",
            "NONE",
        ]
        present_order = [
            p for p in COVERAGE_ORDER if p in cov_summary["coverage_pattern"].values
        ]
        remaining = [
            p for p in cov_summary["coverage_pattern"].values if p not in present_order
        ]
        full_order = present_order + remaining

        cov_summary["coverage_pattern"] = pd.Categorical(
            cov_summary["coverage_pattern"], categories=full_order, ordered=True
        )
        cov_summary = cov_summary.sort_values("coverage_pattern")

        # col_a, col_b = st.columns(2)

        # with col_a:
        fig_cov = go.Figure()
        fig_cov.add_trace(
            go.Bar(
                x=cov_summary["coverage_pattern"].astype(str),
                y=cov_summary["mrr_grouped"],
                name="Grouped MRR",
                marker_color="#3b82f6",
                text=cov_summary["mrr_grouped"].round(3),
                textposition="outside",
            )
        )
        fig_cov.add_trace(
            go.Scatter(
                x=cov_summary["coverage_pattern"].astype(str),
                y=cov_summary["top1"],
                name="Top-1 Exact",
                mode="lines+markers",
                marker=dict(size=8, color="#f59e0b"),
                line=dict(color="#f59e0b", dash="dash"),
                yaxis="y",
            )
        )
        fig_cov.update_layout(
            title="Grouped MRR and Top-1 by Coverage Pattern",
            yaxis=dict(range=[0, 1.05], title="Score"),
            xaxis_title="Coverage Pattern",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5),
            height=370,
            margin=dict(l=0, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

        # with col_b:
        st.markdown("**Per-pattern breakdown**")
        display_cols = {
            "coverage_pattern": "Pattern",
            "mrr_grouped": "Grouped MRR",
            "top1": "Top-1",
            "top3": "Top-3",
            "n": "N (ELR strings)",
        }
        st.dataframe(
            cov_summary[list(display_cols.keys())]
            .rename(columns=display_cols)
            .style.format(
                {"Grouped MRR": "{:.3f}", "Top-1": "{:.1%}", "Top-3": "{:.1%}"}
            )
            .highlight_max(subset=["Grouped MRR"], color="#bbf7d0")
            .highlight_min(subset=["Grouped MRR"], color="#fee2e2"),
            use_container_width=True,
            height=370,
        )

        st.divider()
        st.markdown("### MRR by Coverage Pattern × Noise Level")

        df_cov_noise = load_primary_coverage_noise()
        heat_data = df_cov_noise[
            (df_cov_noise["corpus_strategy"] == best_cs)
            & (df_cov_noise["model_desc"] == best_md)
            & (df_cov_noise["n_distractors"] == best_nd)
            & (df_cov_noise["coverage_pattern"].isin(selected_patterns))
        ]
        heat_pivot = heat_data.pivot(
            index="coverage_pattern", columns="noise_level", values="mrr_grouped"
        )
        # Reorder rows
        heat_pivot = heat_pivot.reindex(
            [p for p in full_order if p in heat_pivot.index]
        )
        noise_col_order = [
            c for c in ["low", "medium", "high"] if c in heat_pivot.columns
        ]
        heat_pivot = heat_pivot[noise_col_order]

        fig_heat = px.imshow(
            heat_pivot.round(3).T,
            #  color_continuous_scale="Blues",
            zmin=0,
            zmax=1,
            text_auto=".3f",
            labels=dict(y="Noise Level", x="Coverage Pattern", color="Grouped MRR"),
            height=380,
        )
        fig_heat.update_layout(margin=dict(l=0, r=10, t=20, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Darker = higher MRR. High-noise strings with minimal signal (pattern A, noise=high) "
            "represent the hardest retrieval cases."
        )

# ===========================================================================
# TAB 4 — Filter Ablation
# ===========================================================================
with tab4:
    st.markdown("### Post-Retrieval Filter Ablation")
    st.markdown("""
Three conditions evaluated on the best corpus+vectorizer config:
- **No filter:** pure TF-IDF ranked list
- **Oracle filter:** ground-truth method class + specimen used to demote mismatching candidates *(upper bound, not production-achievable)*
- **Brand filter:** instrument brand tokens in the ELR string used to impute method class *(inference-time, production-realistic)*

Target population: `has_method=0` rows where TF-IDF alone cannot distinguish method from token signal.
    """)

    if df_filter is None or df_filter.empty:
        st.warning("Filter ablation results not found.")
    else:
        # Overall MRR by filter condition
        filter_summary = df_filter.copy()

        FILTER_LABELS = {
            "none": "No Filter",
            "oracle": "Oracle Filter",
            "brand": "Brand Filter",
        }
        FILTER_COLORS = {"none": "#94a3b8", "oracle": "#22c55e", "brand": "#3b82f6"}
        filter_summary["label"] = filter_summary["filter_applied"].map(FILTER_LABELS)
        filter_summary["color"] = filter_summary["filter_applied"].map(FILTER_COLORS)

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            fig_filter = go.Figure()
            for _, row in filter_summary.iterrows():
                fig_filter.add_trace(
                    go.Bar(
                        name=row["label"],
                        x=[row["label"]],
                        y=[row["mrr_grouped"]],
                        #  marker_color=row["color"],
                        text=[f"{row['mrr_grouped']:.3f}"],
                        textposition="outside",
                        width=0.4,
                    )
                )
            fig_filter.update_layout(
                title="Overall Grouped MRR by Filter Condition",
                yaxis=dict(range=[0, 1.05], title="Grouped MRR"),
                showlegend=False,
                height=340,
                margin=dict(l=0, r=10, t=50, b=10),
                xaxis={
                    "categoryorder": "array",
                    "categoryarray": ["No Filter", "Brand Filter", "Oracle Filter"],
                },
            )

            st.plotly_chart(fig_filter, use_container_width=True)

        with col_f2:
            st.markdown("**Overall metrics by filter condition**")
            display = filter_summary[
                ["label", "mrr_grouped", "top1", "top3", "n"]
            ].copy()
            display.columns = ["Filter", "Grouped MRR", "Top-1", "Top-3", "N"]
            st.dataframe(
                display.style.format(
                    {"Grouped MRR": "{:.3f}", "Top-1": "{:.1%}", "Top-3": "{:.1%}"}
                ).highlight_max(subset=["Grouped MRR"], color="#bbf7d0"),
                use_container_width=True,
                hide_index=True,
            )

            # Oracle delta
            no_filter_mrr = filter_summary.loc[
                filter_summary["filter_applied"] == "none", "mrr_grouped"
            ].values
            oracle_mrr = filter_summary.loc[
                filter_summary["filter_applied"] == "oracle", "mrr_grouped"
            ].values
            brand_mrr = filter_summary.loc[
                filter_summary["filter_applied"] == "brand", "mrr_grouped"
            ].values

            if len(no_filter_mrr) and len(oracle_mrr) and len(brand_mrr):
                oracle_lift = oracle_mrr[0] - no_filter_mrr[0]
                brand_lift = brand_mrr[0] - no_filter_mrr[0]
                recovery = brand_lift / oracle_lift * 100 if oracle_lift > 0 else 0

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Oracle lift",
                    f"+{oracle_lift:.3f}",
                    help="Upper bound gain from perfect metadata",
                )
                c2.metric(
                    "Brand lift",
                    f"+{brand_lift:.3f}",
                    help="Gain from brand token imputation",
                )
                c3.metric(
                    "Brand recovery %",
                    f"{recovery:.0f}%",
                    help="Brand lift / Oracle lift",
                )

        st.divider()
        st.markdown("### Filter Effect Stratified by `has_method`")
        st.markdown(
            "The filter's value is concentrated on `has_method=0` rows. These are strings without an explicit "
            "method token where TF-IDF cannot distinguish NAAT from antigen codes from analyte signal alone."
        )

        strat = df_filter_method.copy()
        strat["filter_label"] = strat["filter_applied"].map(FILTER_LABELS)
        strat["has_method_label"] = strat["has_method"].map(
            {0: "No method token", 1: "Method token present"}
        )

        fig_strat = px.bar(
            strat,
            x="filter_label",
            y="mrr_grouped",
            color="has_method_label",
            barmode="group",
            text=strat["mrr_grouped"].round(3),
            # color_discrete_map={
            #     "No method token": "#f97316",
            #     "Method token present": "#3b82f6",
            # },
            labels={
                "filter_label": "Filter Condition",
                "mrr_grouped": "Grouped MRR",
                "has_method_label": "",
            },
            height=360,
        )
        fig_strat.update_traces(textposition="outside")
        fig_strat.update_layout(
            yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=10, t=40, b=10),
            xaxis={
                "categoryorder": "array",
                "categoryarray": ["No Filter", "Brand Filter", "Oracle Filter"],
            },
        )
        st.plotly_chart(fig_strat, use_container_width=True)

        st.divider()
        st.markdown("### Filter Effect by Coverage Pattern")

        cov_filter = df_filter_coverage.copy()
        cov_filter["filter_label"] = cov_filter["filter_applied"].map(FILTER_LABELS)

        # Sort patterns by no_filter MRR descending
        pattern_order = (
            cov_filter[cov_filter["filter_applied"] == "none"]
            .sort_values("mrr_grouped", ascending=False)["coverage_pattern"]
            .tolist()
        )

        fig_cov_filter = px.bar(
            cov_filter,
            x="coverage_pattern",
            y="mrr_grouped",
            color="filter_label",
            barmode="group",
            # markers=True,
            category_orders={"coverage_pattern": pattern_order},
            # color_discrete_map=FILTER_LABELS
            # and {
            #     "No Filter": "#94a3b8",
            #     "Oracle Filter": "#22c55e",
            #     "Brand Filter": "#3b82f6",
            # },
            labels={
                "coverage_pattern": "Coverage Pattern",
                "mrr_grouped": "Grouped MRR",
                "filter_label": "Filter",
            },
            height=360,
        )
        fig_cov_filter.update_layout(
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=0, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_cov_filter, use_container_width=True)
        st.caption(
            "**Coverage patterns sorted by no-filter MRR.** Oracle lift concentrates on specimen-containing patterns because TF-IDF, guided by specimen tokens, retrieves specific-system codes and the oracle corrects cases where it retrieved the wrong one. For analyte-only strings, TF-IDF defaults to generic catchall codes (Respiratory System Specimen), which are valid for any specimen type under grouped MRR, leaving nothing for the oracle to correct. The brand filter contributes negligible lift: the population of strings with a model token but no method token is too small to move the aggregate."
        )

# ===========================================================================
# TAB 5 — Simulation & Corpus Geometry
# ===========================================================================
with tab5:
    st.markdown("### Simulation Quality & Corpus Geometry")
    st.markdown(
        "This tab diagnoses the simulation pipeline and the TF-IDF corpus structure. "
    )

    # Attempt to load ELR data — needed for all simulation sections
    try:
        df_elr = load_elr()
        elr_ok = True
    except FileNotFoundError:
        elr_ok = False

    try:
        df_loinc_ref = load_loinc()
        loinc_ok = True
    except (FileNotFoundError, ImportError):
        loinc_ok = False

    if not elr_ok:
        st.warning(
            "`data/processed/elr_simulated.csv` not found. "
            "Run `elr_simulation.py` first."
        )
    st.markdown("---")
    st.markdown("#### A. Noise Distribution")
    st.markdown(
        "Three views of the noise taxonomy applied during ELR simulation: "
        "total noise count per string, noise level category (low / medium / high), "
        "and per-type intensity (corruption / compression / omission). "
        "Categories are defined on input transformations independently of any retrieval "
        "model or corpus - they describe what happened to the string, not the retrieval consequence: \n"
        "- **Corruption:** character-level damage (typos: swap, skip, extra character). "
        "Token identity partially or fully destroyed. Sparse by design as LIS instrument "
        "interfaces populate OBX-3 fields programmatically, not via human entry.  \n"
        "- **Compression:** a signal token replaced with an alternate surface form of the "
        "same semantic entity (e.g. SARS-CoV-2 → COVID-19; RNA → NAA). "
        "Information is present but encoded differently; recoverable by a domain-aware model.  \n"
        "- **Omission:** signal deleted entirely: token replaced with empty string, or "
        "entire component (method, specimen) structurally absent from the template. "
        "Unrecoverable without external metadata.  \n"
        "Interpretation tokens (STATUS, RESULT, FINAL) are appended at 10% probability "
        "but are **not counted** toward any noise dimension — they do not damage, "
        "substitute, or remove signal tokens, and have negligible retrieval impact."
    )

    if elr_ok:
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1:
            st.metric("Total ELR strings", f"{len(df_elr):,}")
        with col_n2:
            st.metric("Mean noise total", f"{df_elr['noise_total'].mean():.2f}")
        with col_n3:
            noise_pct = (df_elr["noise_level"] == "high").mean()
            st.metric("High-noise strings", f"{noise_pct:.1%}")

        gen_noise = st.button("▶ Generate noise audit figure", key="gen_noise")
        if gen_noise:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(
                "Simulation Noise Audit", fontsize=12, fontweight="bold", y=1.02
            )

            sns.histplot(
                df_elr["noise_total"], discrete=True, color="#3498db", ax=axes[0]
            )
            axes[0].set_title("Total Noise Count per ELR String", fontsize=10)
            axes[0].set_xlabel("Tokens / chars altered")

            sns.countplot(
                data=df_elr,
                x="noise_level",
                hue="noise_level",
                order=["low", "medium", "high"],
                palette={"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"},
                legend=False,
                ax=axes[1],
            )
            axes[1].set_title("Noise Level Distribution", fontsize=10)
            axes[1].set_xlabel("")

            noise_melt = df_elr[
                ["noise_corruption", "noise_compression", "noise_omission"]
            ].melt(var_name="type", value_name="count")
            noise_melt["type"] = (
                noise_melt["type"].str.replace("noise_", "").str.title()
            )
            sns.boxplot(
                data=noise_melt,
                x="type",
                y="count",
                hue="type",
                palette="Set2",
                legend=False,
                ax=axes[2],
                order=["Corruption", "Compression", "Omission"],
            )
            axes[2].set_title("Per-Type Noise Intensity", fontsize=10)
            axes[2].set_xlabel("")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ------------------------------------------------------------------
        # Noise × coverage pattern independence check
        # ------------------------------------------------------------------
        st.markdown("##### Noise Independence Check")
        st.caption(
            "Mean total noise by coverage pattern × noise level. "
            "If low-information patterns (A, A+S) cluster in the low-noise column, "
            "the simulation conflates information content with noise, a design flaw. "
            "Uniform distribution across columns confirms the two difficulty axes are orthogonal."
        )
        pivot = df_elr.pivot_table(
            index="coverage_pattern",
            columns="noise_level",
            values="noise_total",
            aggfunc="mean",
        ).reindex(columns=["low", "medium", "high"])
        order_pats = df_elr["coverage_pattern"].value_counts().index
        pivot = pivot.reindex([p for p in order_pats if p in pivot.index]).round(2)

        fig_heat = px.imshow(
            pivot.T,
            color_continuous_scale="YlOrRd",
            zmin=0,
            text_auto=True,
            labels=dict(x="Coverage Pattern", y="Noise Level", color="Mean noise"),
            height=280,
        )
        fig_heat.update_layout(margin=dict(l=0, r=10, t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        # ------------------------------------------------------------------
        # Component presence by noise type — confounding analysis
        # ------------------------------------------------------------------
        st.markdown("##### Component Presence by Noise Type")
        st.markdown(
            "The heatmaps below show mean presence of each structural component "
            "(method, specimen, model) broken down by count for each noise type. "
            "A confound exists when a noise type's count correlates with component "
            "presence, meaning the noise axis is not an independent difficulty dimension "
            "but is instead tracking structural changes in the ELR string."
        )

        tab_comp, tab_omit, tab_corrupt = st.tabs(
            ["Compression", "Omission", "Corruption"]
        )

        with tab_comp:
            pivot_comp = (
                df_elr.groupby("noise_compression")[
                    ["has_method", "has_specimen", "has_model"]
                ]
                .mean()
                .round(3)
            )
            pivot_comp.index.name = "Compression count"
            pivot_comp.columns = ["Has Method", "Has Specimen", "Has Model"]

            fig_comp = px.imshow(
                pivot_comp.T,
                color_continuous_scale="YlOrRd",
                zmin=0,
                zmax=1,
                text_auto=True,
                labels=dict(
                    x="Compression count",
                    y="Component",
                    color="Mean presence",
                ),
                height=260,
            )
            fig_comp.update_layout(margin=dict(l=0, r=10, t=10, b=10))
            st.plotly_chart(fig_comp, use_container_width=True)

            st.info(
                "**Note on compression and method signal.** "
                "Compression events substitute analyte tokens with alternate surface forms "
                "of the same semantic entity (e.g. RNA → NAA, PCR). Some of these replacements "
                "happen to also be method tokens. For example, NAA appears in LOINC long common names as "
                "a method descriptor ('by NAA with probe detection'). This means compression "
                "events can incidentally introduce method signal into strings that originally "
                "lacked it, which may partially inflate MRR for higher compression counts. "
                "This is a known limitation of the simulation design, i.e. analyte substitution "
                "and method signal introduction are not fully decoupled. "
                "Compression results should be interpreted alongside coverage patterns rather "
                "than as a clean independent noise dimension."
            )

        with tab_omit:
            pivot_omit = (
                df_elr.groupby("noise_omission")[
                    ["has_method", "has_specimen", "has_model"]
                ]
                .mean()
                .round(3)
            )
            pivot_omit.index.name = "Omission count"
            pivot_omit.columns = ["Has Method", "Has Specimen", "Has Model"]

            fig_omit = px.imshow(
                pivot_omit.T,
                color_continuous_scale="YlOrRd",
                zmin=0,
                zmax=1,
                text_auto=True,
                labels=dict(
                    x="Omission count",
                    y="Component",
                    color="Mean presence",
                ),
                height=260,
            )
            fig_omit.update_layout(margin=dict(l=0, r=10, t=10, b=10))
            st.plotly_chart(fig_omit, use_container_width=True)

            st.warning(
                "**Omission is the dominant noise dimension but has a structural confound.** "
                "Omission count correlates with method token absence (Pearson r = −0.73). "
                "This occurs because structural template omission (has_method=0) and target "
                "deletion both increment the omission counter, and both reduce method signal. "
                "As a result, high-omission rows are systematically different from low-omission "
                "rows in method presence, not just in noise level. "
                "Omission-stratified MRR results should be interpreted in the context of "
                "coverage patterns rather than as a pure noise effect. "
                "TF-IDF is substantially more robust to omission than sentence transformers. This is "
                "a finding consistent with sparse retrieval's additive independence of token "
                "contributions, where remaining tokens retain full discriminating power "
                "regardless of what was deleted."
            )

        with tab_corrupt:
            pivot_corrupt = (
                df_elr.groupby("noise_corruption")[
                    ["has_method", "has_specimen", "has_model"]
                ]
                .mean()
                .round(3)
            )
            pivot_corrupt.index.name = "Corruption count"
            pivot_corrupt.columns = ["Has Method", "Has Specimen", "Has Model"]

            fig_corrupt = px.imshow(
                pivot_corrupt.T,
                color_continuous_scale="YlOrRd",
                zmin=0,
                zmax=1,
                text_auto=True,
                labels=dict(
                    x="Corruption count",
                    y="Component",
                    color="Mean presence",
                ),
                height=260,
            )
            fig_corrupt.update_layout(margin=dict(l=0, r=10, t=10, b=10))
            st.plotly_chart(fig_corrupt, use_container_width=True)

            st.success(
                "**Corruption is largely unconfounded with component presence.** "
                "Character-level typo injection operates on the assembled ELR string "
                "after structural components are in place — it does not add or remove "
                "components, only degrades individual tokens. "
                "Component presence rates are therefore stable across corruption counts. "
                "Corruption counts above 1 represent very small populations (n < 30) "
                "given the sparse typo injection rate (10% base, aggressive decay), "
                "reflecting the realistic LIS setting where instrument interfaces "
                "populate fields programmatically with low character-level error rates. "
                "Results for corruption count ≥ 2 should be interpreted cautiously "
                "due to small sample size."
            )
    # -----------------------------------------------------------------------
    # SECTION B — Structural composition
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### B. ELR Structural Composition")
    st.markdown(
        "Component prevalence (what fraction of simulated strings contain each signal type) "
        "and coverage pattern frequency. These reflect the template weight distribution "
        "and confirm the simulation produces the intended mix of easy and hard retrieval cases."
    )

    if elr_ok:
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            comp_cols = ["has_analyte", "has_method", "has_specimen", "has_model"]
            comp_labels_map = {
                "has_analyte": "Analyte",
                "has_method": "Method",
                "has_specimen": "Specimen",
                "has_model": "Model",
            }
            comp_means = df_elr[comp_cols].mean().reset_index()
            comp_means.columns = ["component", "prevalence"]
            comp_means["label"] = comp_means["component"].map(comp_labels_map)

            fig_comp = px.bar(
                comp_means,
                x="label",
                y="prevalence",
                color="label",
                # color_discrete_sequence=["#3498db", "#e67e22", "#2ecc71", "#9b59b6"],
                text=comp_means["prevalence"].map(lambda v: f"{v:.1%}"),
                labels={"label": "", "prevalence": "Prevalence"},
                title="Proportion of ELR Strings Containing Component",
                height=320,
            )
            fig_comp.update_traces(textposition="outside", showlegend=False)
            fig_comp.update_layout(
                yaxis=dict(range=[0, 1.12]), margin=dict(l=0, r=0, t=50, b=10)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_s2:
            top_patterns = (
                df_elr["coverage_pattern"].value_counts().head(12).reset_index()
            )
            top_patterns.columns = ["pattern", "count"]
            fig_pat = px.bar(
                top_patterns,
                x="count",
                y="pattern",
                orientation="h",
                # color_discrete_sequence=["#3498db"],
                labels={"pattern": "Coverage Pattern", "count": "Count"},
                title="Top Coverage Patterns (by frequency)",
                height=320,
            )
            fig_pat.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                margin=dict(l=0, r=0, t=50, b=10),
            )
            st.plotly_chart(fig_pat, use_container_width=True)

        st.markdown("##### Token Count by Corruption Count")
        st.caption(
            "Checks whether character-level corruption (typos) is confounded "
            "with string length. Corruption events operate on the assembled string "
            "without adding or removing tokens, so token count should be stable "
            "across corruption counts. Note that noise_level is not used here because "
            "it aggregates all three noise types resulting in high noise_level strings "
            "being shorter by construction due to omission events."
        )
        fig_len = px.box(
            df_elr,
            x="noise_corruption",
            y="analyte_len",
            category_orders={
                "noise_corruption": sorted(df_elr["noise_corruption"].unique().tolist())
            },
            color="noise_corruption",
            labels={
                "noise_corruption": "Corruption count (typos)",
                "analyte_len": "Token count",
            },
            height=300,
        )
        fig_len.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=10))
        st.plotly_chart(fig_len, use_container_width=True)
        st.caption(
            "Stable token count across corruption counts confirms that typo "
            "injection does not systematically alter string length, hence corruption "
            "is independent of structural composition."
        )

    # -----------------------------------------------------------------------
    # SECTION C — Token frequency: corpus vs ELR
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### C. Vocabulary Overlap: LOINC Corpus vs ELR Queries")
    st.markdown(
        "Token frequency comparison between the expanded LOINC corpus (index side) "
        "and normalized ELR strings (query side). "
        "Divergences — tokens common on one side but absent on the other — "
        "point to genuine vocabulary gaps that TF-IDF cannot bridge without expansion."
    )

    if elr_ok and loinc_ok:
        from collections import Counter
        from src.clinical_utils import clean_text as _clean_text
        from src.model_building_utils import build_corpus as _build_corpus

        @st.cache_data
        def _token_freq(series_list, top_n=30):
            tokens = " ".join(
                pd.Series(series_list).fillna("").map(_clean_text)
            ).split()
            return pd.DataFrame(
                Counter(tokens).most_common(top_n), columns=["token", "count"]
            )

        sel_strategy_tok = st.selectbox(
            "Corpus strategy for token frequency comparison",
            options=list(CORPUS_LABELS.keys()),
            index=list(CORPUS_LABELS.keys()).index("lcn_method_dict_combined"),
            format_func=lambda x: CORPUS_LABELS.get(x, x),
            key="tok_strategy",
        )

        if st.button("▶ Generate token frequency comparison", key="gen_tok"):
            rn_sw = load_rn_stopwords()
            corpus_series = _build_corpus(df_loinc_ref, sel_strategy_tok, rn_sw)
            elr_col = (
                "elr_name_normalized"
                if "elr_name_normalized" in df_elr.columns
                else "elr_name"
            )

            corpus_freq = _token_freq(corpus_series.tolist(), top_n=25)
            elr_freq = _token_freq(df_elr[elr_col].tolist(), top_n=25)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                fig_c = px.bar(
                    corpus_freq.iloc[::-1],
                    x="count",
                    y="token",
                    orientation="h",
                    color_discrete_sequence=["#2980b9"],
                    title=f"LOINC Corpus ({CORPUS_LABELS.get(sel_strategy_tok, sel_strategy_tok)})",
                    labels={"token": "", "count": "Token count"},
                    height=500,
                )
                fig_c.update_layout(margin=dict(l=0, r=0, t=50, b=10))
                st.plotly_chart(fig_c, use_container_width=True)
            with col_t2:
                fig_e = px.bar(
                    elr_freq.iloc[::-1],
                    x="count",
                    y="token",
                    orientation="h",
                    color_discrete_sequence=["#c0392b"],
                    title="ELR Strings (normalized, query side)",
                    labels={"token": "", "count": "Token count"},
                    height=500,
                )
                fig_e.update_layout(margin=dict(l=0, r=0, t=50, b=10))
                st.plotly_chart(fig_e, use_container_width=True)

    # -----------------------------------------------------------------------
    # SECTION D — Within/between group similarity
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### D. Within- vs Between-Group Corpus Discriminability")
    st.markdown(
        "For each corpus strategy: mean cosine similarity of LOINC code pairs that share "
        "the same `method` (within-group) vs pairs that differ (between-group). "
        "**Cohen's d** (pooled-std normalized) is the cross-strategy-comparable metric.  \n "
        "d ≥ 0.5 is medium separation, d ≥ 0.8 is large. "
        "Use the sidebar to pick which strategies to compare."
    )

    if loinc_ok and selected_geo_strategies:
        with st.spinner("Computing pairwise cosine similarities…"):
            try:
                wb_df = compute_within_between(
                    tuple(selected_geo_strategies), group_by="method_typ"
                )

                # col_wb1, col_wb2 = st.columns(2)

                # with col_wb1:
                fig_wb = go.Figure()
                fig_wb.add_trace(
                    go.Bar(
                        name="Within group",
                        x=wb_df["corpus_label"],
                        y=wb_df["within"],
                        # marker_color="#2980b9",
                        text=wb_df["within"].round(3),
                        textposition="outside",
                    )
                )
                fig_wb.add_trace(
                    go.Bar(
                        name="Between group",
                        x=wb_df["corpus_label"],
                        y=wb_df["between"],
                        #  marker_color="#e74c3c",
                        text=wb_df["between"].round(3),
                        textposition="outside",
                    )
                )
                fig_wb.update_layout(
                    barmode="group",
                    title="Mean Cosine Similarity Within vs Between Method Groups",
                    yaxis=dict(title="Mean cosine similarity", range=[0, 1.05]),
                    legend=dict(orientation="h", yanchor="bottom", y=0.8, x=0.5),
                    height=380,
                    margin=dict(l=0, r=0, t=60, b=80),
                )
                fig_wb.update_xaxes(tickangle=25)
                st.plotly_chart(fig_wb, use_container_width=True)

                # with col_wb2:
                fig_d = go.Figure()
                fig_d.add_trace(
                    go.Bar(
                        x=wb_df["corpus_label"],
                        y=wb_df["cohens_d"],
                        # marker_color="#8e44ad",
                        text=wb_df["cohens_d"].round(3),
                        textposition="outside",
                    )
                )
                for thresh, label in [
                    (0.2, "small"),
                    (0.5, "medium"),
                    (0.8, "large"),
                ]:
                    fig_d.add_hline(
                        y=thresh,
                        line_dash="dot",
                        line_color="dark grey",
                        annotation_text=label,
                        annotation_position="top right",
                    )
                fig_d.update_layout(
                    title="Cohen's d — Discriminability by Corpus Strategy",
                    yaxis=dict(title="Cohen's d", rangemode="tozero"),
                    showlegend=False,
                    height=380,
                    margin=dict(l=0, r=0, t=60, b=80),
                )
                fig_d.update_xaxes(tickangle=25)
                st.plotly_chart(fig_d, use_container_width=True)

                st.dataframe(
                    wb_df[["corpus_label", "within", "between", "cohens_d", "ratio"]]
                    .rename(
                        columns={
                            "corpus_label": "Corpus Strategy",
                            "within": "Within-group sim",
                            "between": "Between-group sim",
                            "cohens_d": "Cohen's d",
                            "ratio": "Ratio (w/b)",
                        }
                    )
                    .style.format(
                        {
                            "Within-group sim": "{:.3f}",
                            "Between-group sim": "{:.3f}",
                            "Cohen's d": "{:.3f}",
                            "Ratio (w/b)": "{:.3f}",
                        }
                    )
                    .highlight_max(subset=["Cohen's d"], color="#bbf7d0")
                    .highlight_min(subset=["Cohen's d"], color="#fee2e2"),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "Sorted by Cohen's d descending. Higher d = the corpus separates NAAT "
                    "from antigen codes more cleanly in embedding space."
                )
            except ImportError as e:
                st.warning(
                    f"Could not import project source modules: `{e}`.  \n"
                    "Make sure `src/` is on your Python path (run `streamlit run app.py` "
                    "from the project root)."
                )
    elif not selected_geo_strategies:
        st.info("Select at least one strategy in the sidebar to compute this section.")

    # -----------------------------------------------------------------------
    # SECTION E — UMAP corpus geometry (requires umap-learn)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### E. UMAP Corpus Geometry")
    st.markdown(
        "Projects each LOINC code's TF-IDF vector into 2D using UMAP (cosine metric). "
        "Tight method-class clusters confirm the corpus separates NAAT and antigen codes "
        "in embedding space. "
        # "Requires `umap-learn` (`pip install umap-learn`)."
    )

    if loinc_ok:
        umap_strategy = st.selectbox(
            "Corpus strategy for UMAP",
            options=list(CORPUS_LABELS.keys()),
            index=list(CORPUS_LABELS.keys()).index("lcn_method_dict_combined"),
            format_func=lambda x: CORPUS_LABELS.get(x, x),
            key="umap_strategy",
        )
        umap_color = st.radio(
            "Color by",
            ["method_typ", "system", "method_class"],
            horizontal=True,
            key="umap_color",
        )

        if st.button("▶ Generate UMAP (~15–30s)", key="gen_umap"):
            try:
                import umap  # noqa
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import normalize
                from src.model_building_utils import build_corpus as _build_corpus2

                rn_sw = load_rn_stopwords()
                corpus = _build_corpus2(df_loinc_ref, umap_strategy, rn_sw)
                vec = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 1),
                    sublinear_tf=True,
                    min_df=1,
                    max_df=0.85,
                )
                matrix = normalize(vec.fit_transform(corpus))

                reducer = umap.UMAP(
                    n_components=2,
                    metric="cosine",
                    n_neighbors=12,
                    min_dist=0.1,
                    random_state=42,
                )
                emb = reducer.fit_transform(matrix)

                color_col = (
                    umap_color if umap_color in df_loinc_ref.columns else "method_typ"
                )
                labels_umap = (
                    df_loinc_ref[color_col].fillna("unknown").astype(str).values
                )
                unique_labels_umap = sorted(set(labels_umap))
                palette_umap = sns.color_palette("tab10", len(unique_labels_umap))
                cmap_umap = dict(zip(unique_labels_umap, palette_umap))

                fig_umap, ax = plt.subplots(figsize=(10, 7))
                for lab in unique_labels_umap:
                    mask = labels_umap == lab
                    ax.scatter(
                        emb[mask, 0],
                        emb[mask, 1],
                        label=lab,
                        color=cmap_umap[lab],
                        s=80,
                        alpha=0.85,
                        edgecolors="white",
                        linewidth=0.4,
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"UMAP — {CORPUS_LABELS.get(umap_strategy, umap_strategy)}, "
                    f"colored by {color_col}",
                    fontsize=11,
                )
                ax.legend(
                    title=color_col,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    fontsize=8,
                    framealpha=0.8,
                )
                plt.tight_layout()
                st.pyplot(fig_umap, use_container_width=True)
                plt.close(fig_umap)

            except ImportError:
                st.error(
                    "`umap-learn` is not installed. Run `pip install umap-learn` and restart."
                )


# ===========================================================================
# TAB 6 — TF-IDF vs Sentence Transformers
# ===========================================================================
with tab6:
    st.markdown("### TF-IDF vs Sentence Transformers")
    st.markdown(
        "Both approaches were evaluated on the same simulated ELR validation set. "
        "TF-IDF used the best config from the primary ablation (`lcn_method_dict_combined`, "
        "word unigrams). ST models were evaluated under two corpus conditions: "
        "**regular** (long common name only) and **boosted** (LCN + appended component, "
        "method_typ, and system fields). The ST corpus deliberately uses natural language form without token expansion, since dense encoders derive meaning from syntactic structure rather than lexical overlap."
    )

    # -----------------------------------------------------------------------
    # SECTION A — Headline comparison
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### A · Overall Performance")

    rows = []
    for _, row in df_st.iterrows():
        rows.append(
            {
                "Model": ST_MODEL_LABELS.get(row["model_type"], row["model_type"]),
                "Corpus": STRATEGY_LABELS.get(row["strategy"], row["strategy"]),
                "Grouped MRR": round(row["mrr_grouped"], 3),
                "Top-1": round(row["top1"], 3),
                "Top-3": round(row["top3"], 3),
            }
        )

    reg = df_st[df_st["strategy"] == "regular_corpus"]
    bst = df_st[df_st["strategy"] == "boosted_corpus"]

    # Add TF-IDF reference row
    tfidf_row = {
        "Model": "✦ TF-IDF (best config)",
        "Corpus": "lcn_method_dict_combined",
        "Grouped MRR": TFIDF_REF["mrr_grouped"],
        "Top-1": TFIDF_REF["top1"],
        "Top-3": TFIDF_REF["top3"],
    }
    df_compare = pd.DataFrame([tfidf_row] + rows)

    st.dataframe(
        df_compare.style.format(
            {"Grouped MRR": "{:.3f}", "Top-1": "{:.3f}", "Top-3": "{:.3f}"}
        )
        .highlight_max(subset=["Grouped MRR"], color="#bbf7d0")
        .highlight_min(subset=["Grouped MRR"], color="#fee2e2")
        .apply(
            lambda col: [
                "font-weight: bold; background-color: #eff6ff"
                if v == "✦ TF-IDF (best config)"
                else ""
                for v in df_compare["Model"]
            ],
            subset=["Model"],
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Green = best grouped MRR. Red = worst. TF-IDF row highlighted in blue. "
        "All ST results on validation set, regular_corpus = LCN only, "
        "boosted_corpus = LCN + component + method_typ + system appended as text."
    )

    # -----------------------------------------------------------------------
    # SECTION B — Boosted vs regular corpus effect
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### B · Effect of Appending Structured LOINC Fields to the Corpus")
    st.markdown(
        "The boosted corpus appends `component`, `method_typ`, and `system` as plain text "
        "to the long common name before encoding. For TF-IDF this is analogous to the "
        "`lcn_method_dict_combined` strategy and helps considerably. For ST models the "
        "effect is **strongly model dependent** as most models are hurt by the additional "
        "structured text, while SapBERT variants benefit. This reveals a fundamental "
        "difference in how these models use input text."
    )

    reg_mrr = reg.set_index("model_type")["mrr_grouped"].round(3)
    bst_mrr = bst.set_index("model_type")["mrr_grouped"].round(3)
    delta_df = pd.DataFrame(
        {
            "model_label": [ST_MODEL_LABELS.get(m, m) for m in reg_mrr.index],
            "Regular": reg_mrr.values,
            "Boosted": bst_mrr.reindex(reg_mrr.index).values,
        }
    )
    delta_df["Delta"] = (delta_df["Boosted"] - delta_df["Regular"]).round(3)
    delta_df = delta_df.sort_values("Delta", ascending=False)

    # col_b1, col_b2 = st.columns(2)

    # with col_b1:
    fig_boost = go.Figure()
    fig_boost.add_trace(
        go.Bar(
            name="Regular (LCN only)",
            x=delta_df["model_label"],
            y=delta_df["Regular"],
            # marker_color="#3b82f6",
            text=delta_df["Regular"],
            textposition="outside",
        )
    )
    fig_boost.add_trace(
        go.Bar(
            name="Boosted (LCN + axes)",
            x=delta_df["model_label"],
            y=delta_df["Boosted"],
            #  marker_color="#f97316",
            text=delta_df["Boosted"],
            textposition="outside",
        )
    )
    fig_boost.add_hline(
        y=TFIDF_REF["mrr_grouped"],
        line_dash="dash",
        # line_color="#22c55e",
        annotation_text=f"TF-IDF ({TFIDF_REF['mrr_grouped']})",
        annotation_position="top left",
    )
    fig_boost.update_layout(
        barmode="group",
        title="Grouped MRR: Regular vs Boosted Corpus",
        yaxis=dict(range=[0, 0.85], title="Grouped MRR"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5),
        height=400,
        margin=dict(l=0, r=0, t=60, b=100),
    )
    fig_boost.update_xaxes(tickangle=30)
    st.plotly_chart(fig_boost, use_container_width=True)

    # with col_b2:
    colors = ["#22c55e" if d > 0 else "#ef4444" for d in delta_df["Delta"]]
    fig_delta = go.Figure(
        go.Bar(
            x=delta_df["model_label"],
            y=delta_df["Delta"],
            marker_color=colors,
            text=delta_df["Delta"].map(lambda v: f"+{v:.3f}" if v > 0 else f"{v:.3f}"),
            textposition="outside",
        )
    )
    fig_delta.add_hline(y=0, line_color="#64748b", line_width=1)
    fig_delta.update_layout(
        title="MRR Delta (Boosted − Regular)",
        yaxis=dict(title="Delta grouped MRR"),
        showlegend=False,
        height=400,
        margin=dict(l=0, r=0, t=60, b=100),
    )
    fig_delta.update_xaxes(tickangle=30)
    st.plotly_chart(fig_delta, use_container_width=True)

    st.markdown(
        "**Why does boosting hurt most ST models?** "
        "General-purpose sentence encoders (MiniLM, msmarco-distilbert, S-PubMedBert-MS-MARCO) "
        "were trained on natural language pairs. Appending structured fields like "
        "`method Probe.amp.tar system Nph` introduces out-of-distribution token sequences "
        "that fragment the semantic embedding rather than enriching it, the model encodes "
        "the structured text literally rather than understanding it. "
        "SapBERT is the exception: trained on biomedical entity synonym pairs, it can "
        "leverage the structured field names as additional entity context, so boosting helps."
    )

    # -----------------------------------------------------------------------
    # SECTION C — Coverage pattern head-to-head
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### C · Where ST Beats TF-IDF: Coverage Pattern Head-to-Head")
    st.markdown(
        "Comparing the best ST model (S-PubMedBert-MS-MARCO, regular corpus) against "
        "TF-IDF by coverage pattern reveals a consistent structural advantage for ST "
        "on **interpretation token strings** (patterns containing `I`), primarily due to these non-informative tokens diluting the TF values of other tokens in the query."
        "TF-IDF dominates on all other patterns, particularly those with analyte and method tokens."
    )

    # Coverage pattern comparison — best ST model (regular) vs TF-IDF
    tfidf_cov = (
        df_filter_coverage[df_filter_coverage["filter_applied"] == "none"]
        .set_index("coverage_pattern")["mrr_grouped"]
        .round(3)
    )

    best_st_model = "pritamdeka/S-PubMedBert-MS-MARCO"
    st_cov = (
        df_st_coverage[
            (df_st_coverage["model_type"] == best_st_model)
            & (df_st_coverage["strategy"] == "regular_corpus")
        ]
        .set_index("coverage_pattern")["mrr_grouped"]
        .round(3)
    )

    cov_df = pd.DataFrame(
        {
            "TF-IDF": tfidf_cov,
            "S-PubMedBert (regular)": st_cov,
        }
    ).dropna()
    cov_df["Delta (ST − TF-IDF)"] = (
        cov_df["S-PubMedBert (regular)"] - cov_df["TF-IDF"]
    ).round(3)
    cov_df = cov_df.sort_values("Delta (ST − TF-IDF)", ascending=False)

    # col_c1, col_c2 = st.columns(2)

    # with col_c1:
    fig_cov = go.Figure()
    fig_cov.add_trace(
        go.Bar(
            name="TF-IDF",
            x=cov_df.index,
            y=cov_df["TF-IDF"],
            marker_color="#3b82f6",
        )
    )
    fig_cov.add_trace(
        go.Bar(
            name="S-PubMedBert (regular)",
            x=cov_df.index,
            y=cov_df["S-PubMedBert (regular)"],
            marker_color="#f97316",
        )
    )
    fig_cov.update_layout(
        barmode="group",
        title="Grouped MRR by Coverage Pattern",
        yaxis=dict(range=[0, 1.0], title="Grouped MRR"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5),
        height=380,
        margin=dict(l=0, r=0, t=60, b=80),
    )
    fig_cov.update_xaxes(tickangle=30)
    st.plotly_chart(fig_cov, use_container_width=True)

    # with col_c2:
    delta_colors = [
        "#22c55e" if d > 0 else "#ef4444" for d in cov_df["Delta (ST − TF-IDF)"]
    ]
    fig_delta_cov = go.Figure(
        go.Bar(
            x=cov_df.index,
            y=cov_df["Delta (ST − TF-IDF)"],
            marker_color=delta_colors,
            text=cov_df["Delta (ST − TF-IDF)"].map(
                lambda v: f"+{v:.3f}" if v > 0 else f"{v:.3f}"
            ),
            textposition="outside",
        )
    )
    fig_delta_cov.add_hline(y=0, line_color="#64748b", line_width=1)
    fig_delta_cov.update_layout(
        title="Delta (S-PubMedBert − TF-IDF) by Coverage Pattern",
        yaxis=dict(title="Delta grouped MRR"),
        showlegend=False,
        height=380,
        margin=dict(l=0, r=0, t=60, b=80),
    )
    fig_delta_cov.update_xaxes(tickangle=30)
    st.plotly_chart(fig_delta_cov, use_container_width=True)

    st.markdown(
        "**Why does ST win on I-patterns?** "
        "Strings with interpretation tokens (`RESULT`, `FINAL`, `INTERP`) but weak analyte "
        "or method signal are nearly opaque to TF-IDF, acting as noise, as there are no discriminating "
        "n-grams to match against the LOINC corpus. ST can leverage semantic context from "
        "surrounding tokens to infer the likely test type."
    )

    # -----------------------------------------------------------------------
    # SECTION D — Noise robustness
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### D · Noise Robustness")
    st.markdown(
        "TF-IDF is remarkably stable across noise levels as the explicit method token dictionary "
        "provides robust signal even under high character corruption. Most ST models degrade "
        "under high noise, with MiniLM showing the sharpest drop. "
        "S-PubMedBert-MS-MARCO is the most noise robust ST model."
    )

    noise_rows = []
    for model_type, label in ST_MODEL_LABELS.items():
        sub = (
            df_st_noise[
                (df_st_noise["model_type"] == model_type)
                & (df_st_noise["strategy"] == "regular_corpus")
            ]
            .set_index("noise_level")["mrr_grouped"]
            .round(3)
        )
        noise_rows.append(
            {
                "Model": label,
                "Low noise": sub.get("low", None),
                "Medium noise": sub.get("medium", None),
                "High noise": sub.get("high", None),
                "High−Low": round(sub.get("high", 0) - sub.get("low", 0), 3),
            }
        )

    # Add TF-IDF reference
    noise_rows.insert(
        0,
        {
            "Model": "✦ TF-IDF (best config)",
            "Low noise": TFIDF_REF["noise"]["low"],
            "Medium noise": TFIDF_REF["noise"]["medium"],
            "High noise": TFIDF_REF["noise"]["high"],
            "High−Low": round(
                TFIDF_REF["noise"]["high"] - TFIDF_REF["noise"]["low"], 3
            ),
        },
    )
    noise_df = pd.DataFrame(noise_rows)

    col_d1, col_d2 = st.columns([1.4, 1])
    with col_d1:
        noise_plot = noise_df.melt(
            id_vars="Model",
            value_vars=["Low noise", "Medium noise", "High noise"],
            var_name="Noise Level",
            value_name="Grouped MRR",
        )
        fig_noise = px.line(
            noise_plot,
            x="Noise Level",
            y="Grouped MRR",
            color="Model",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
            category_orders={
                "Noise Level": ["Low noise", "Medium noise", "High noise"]
            },
            height=360,
        )
        fig_noise.update_layout(
            yaxis=dict(range=[0, 0.85]),
            legend=dict(orientation="v", x=1.02, y=1),
            margin=dict(l=0, r=0, t=20, b=10),
        )
        st.plotly_chart(fig_noise, use_container_width=True)

    with col_d2:
        st.markdown("**High−Low delta** (negative = degrades under noise)")
        st.dataframe(
            noise_df[["Model", "Low noise", "Medium noise", "High noise", "High−Low"]]
            .style.format(
                {
                    "Low noise": "{:.3f}",
                    "Medium noise": "{:.3f}",
                    "High noise": "{:.3f}",
                    "High−Low": "{:+.3f}",
                }
            )
            .highlight_max(subset=["High−Low"], color="#bbf7d0")
            .highlight_min(subset=["High−Low"], color="#fee2e2"),
            hide_index=True,
            use_container_width=True,
            height=300,
        )

    # -----------------------------------------------------------------------
    # SECTION E — has_method stratification
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### E · Method Token Presence")
    st.markdown(
        "Neither approach fully solves the `has_method=0` case, but the gap is "
        "narrower for TF-IDF than most ST models. Moreover, ST still lags behind TF-IDF "
        "on both strata, as semantic embeddings do not recover the method signal "
        "that TF-IDF extracts from explicit method tokens."
    )

    meth_rows = []
    for model_type, label in ST_MODEL_LABELS.items():
        sub = (
            df_st_method[
                (df_st_method["model_type"] == model_type)
                & (df_st_method["strategy"] == "regular_corpus")
            ]
            .set_index("has_method")["mrr_grouped"]
            .round(3)
        )
        meth_rows.append(
            {
                "Model": label,
                "has_method=0": sub.get(0, None),
                "has_method=1": sub.get(1, None),
            }
        )
    meth_rows.insert(
        0,
        {
            "Model": "✦ TF-IDF (best config)",
            "has_method=0": TFIDF_REF["has_method"][0],
            "has_method=1": TFIDF_REF["has_method"][1],
        },
    )
    meth_df = pd.DataFrame(meth_rows)

    meth_plot = meth_df.melt(
        id_vars="Model",
        value_vars=["has_method=0", "has_method=1"],
        var_name="Method token",
        value_name="Grouped MRR",
    )
    fig_meth = px.bar(
        meth_plot,
        x="Model",
        y="Grouped MRR",
        color="Method token",
        barmode="group",
        color_discrete_map={
            "has_method=0": "#f97316",
            "has_method=1": "#3b82f6",
        },
        height=360,
    )
    fig_meth.update_layout(
        yaxis=dict(range=[0, 0.85]),
        xaxis_tickangle=30,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=100),
    )
    st.plotly_chart(fig_meth, use_container_width=True)

    # -----------------------------------------------------------------------
    # SECTION F — Summary: complementary failure modes
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### F · Summary: Complementary Failure Modes")
    st.info(
        "**TF-IDF dominates overall** (0.735 vs 0.617 best ST grouped MRR) because the "
        "retrieval signal is terminological (abbreviation and token matching between ELR strings and "
        "LOINC vocabulary) which explicit method token dictionaries handle precisely.  \n\n"
        "**ST shows a relative advantage on I-pattern strings** (+0.128 to +0.365 "
        "over TF-IDF on patterns containing interpretation tokens), where tokens like 'RESULT' and 'FINAL' dilute the term frequency of discriminative signal tokens in TF-IDF queries. ST is less sensitive to this dilution because it encodes the full sequence rather than a bag of token counts. This is not recovered by the TF-IDF brand filter, which addresses method imputation rather than noise sensitivity.  \n\n"
        "**Boosting hurts most ST models** as appending structured LOINC field text fragments "
        "semantic embeddings for general purpose encoders. SapBERT is the exception, benefiting "
        "from the additional entity context due to its biomedical synonym training objective.  \n\n"
        "**Noise robustness favours TF-IDF** as explicit token matching is stable across noise "
        "levels. MiniLM degrades sharply under high noise (high−low = +0.197 for TF-IDF vs "
        "−0.197 for MiniLM), limiting its reliability in realistic ELR conditions.  \n\n"
        "**Natural next step**: a hybrid or ensemble approach, TF-IDF for method rich strings, "
        "ST for interpretation only strings, could combine both strengths without their respective weaknesses."
    )


# ===========================================================================
# TAB 7 — Test Set Results
# ===========================================================================
with tab7:
    st.markdown("### Test Set Evaluation")
    st.markdown(
        "The simulated ELRs were split into val and test sets stratified by the LOINC codes to ensure similar distributions (since no model training is done, a train split is not required). All configs are fixed from the validation ablation. This tab "
        "confirms generalization, not selecting anything new."
    )
    st.info(
        "**Fixed config:** `lcn_method_dict_combined` · word unigrams · 0 distractors  \n"
        "Run `notebooks/test_set_evaluation.ipynb` to generate `data/results/test_filter_ablation.csv`."
    )

    df_test_all, test_data_loaded = load_test_results()

    if not test_data_loaded or df_test_all.empty:
        st.warning(
            "`data/results/test_filter_ablation.csv` not found.  \n"
            "Run **`notebooks/05_test_set_evaluation.ipynb`** to generate it, then restart the app."
        )
        st.stop()

    # -----------------------------------------------------------------------
    # SECTION A — Headline: val vs test
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### A · Headline: Validation vs Test")

    CONDITION_ORDER_T = ["no_filter", "oracle_filter", "brand_filter"]
    CONDITION_LABELS_T = {
        "no_filter": "No Filter (production config)",
        "oracle_filter": "Oracle Filter (upper bound)",
        "brand_filter": "Brand Filter (production-feasible)",
    }

    val_headline = (
        df_filter[["filter_applied", "mrr_grouped", "top1", "top3", "top5"]]
        .set_index("filter_applied")
        .rename(columns=lambda c: f"val_{c}")
        .rename_axis("filter_condition")
    )
    test_headline = (
        df_test_all.groupby("filter_condition")[["mrr_grouped", "top1", "top3", "top5"]]
        .mean()
        .rename(columns=lambda c: f"test_{c}")
    )
    # Reindex requires mapping between the keys if they differ.
    # But since val_headline is keyed by filter_applied ('none', 'oracle', 'brand')
    # and test_headline is keyed by filter_condition ('no_filter', 'oracle_filter', 'brand_filter')
    # We should map val_headline's index to match test_headline before join.
    val_headline.index = val_headline.index.map(
        {"none": "no_filter", "oracle": "oracle_filter", "brand": "brand_filter"}
    )
    headline = val_headline.join(test_headline).reindex(CONDITION_ORDER_T)
    headline["gap_mrr"] = (
        headline["test_mrr_grouped"] - headline["val_mrr_grouped"]
    ).round(4)
    headline.index = [CONDITION_LABELS_T.get(i, i) for i in headline.index]

    nf_val = df_filter.loc[df_filter["filter_applied"] == "none", "mrr_grouped"].values[
        0
    ]
    nf_test = df_test_all[df_test_all["filter_condition"] == "no_filter"][
        "mrr_grouped"
    ].mean()
    nf_top1_test = df_test_all[df_test_all["filter_condition"] == "no_filter"][
        "top1"
    ].mean()
    nf_top3_test = df_test_all[df_test_all["filter_condition"] == "no_filter"][
        "top3"
    ].mean()
    nf_gap = nf_test - nf_val

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Test Grouped MRR",
        f"{nf_test:.3f}",
        delta=f"{nf_gap:+.3f} vs val",
        delta_color="normal",
        help="No-filter condition, held-out test set",
    )
    c2.metric(
        "Val Grouped MRR", f"{nf_val:.3f}", help="No-filter condition, validation set"
    )
    c3.metric("Test Top-1", f"{nf_top1_test:.1%}")
    c4.metric("Test Top-3", f"{nf_top3_test:.1%}")

    st.markdown("**Full headline table - all filter conditions**")
    display_cols = {
        "val_mrr_grouped": "Val MRR",
        "test_mrr_grouped": "Test MRR",
        "gap_mrr": "Gap (Test−Val)",
        "val_top1": "Val Top-1",
        "test_top1": "Test Top-1",
        "val_top3": "Val Top-3",
        "test_top3": "Test Top-3",
    }
    headline_display = headline[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(
        headline_display.style.format(
            {
                "Val MRR": "{:.3f}",
                "Test MRR": "{:.3f}",
                "Gap (Test−Val)": "{:+.3f}",
                "Val Top-1": "{:.1%}",
                "Test Top-1": "{:.1%}",
                "Val Top-3": "{:.1%}",
                "Test Top-3": "{:.1%}",
            }
        ).applymap(
            lambda v: (
                "color: #16a34a"
                if isinstance(v, float) and v > 0.005
                else ("color: #dc2626" if isinstance(v, float) and v < -0.005 else "")
            ),
            subset=["Gap (Test−Val)"],
        ),
        use_container_width=True,
    )

    bar_data = []
    for cond in CONDITION_ORDER_T:
        map_cond = {
            "no_filter": "none",
            "oracle_filter": "oracle",
            "brand_filter": "brand",
        }
        v = df_filter.loc[
            df_filter["filter_applied"] == map_cond[cond], "mrr_grouped"
        ].values[0]
        t = df_test_all[df_test_all["filter_condition"] == cond]["mrr_grouped"].mean()
        bar_data.append(
            {
                "Condition": CONDITION_LABELS_T.get(cond, cond),
                "Grouped MRR": v,
                "Split": "Validation",
            }
        )
        bar_data.append(
            {
                "Condition": CONDITION_LABELS_T.get(cond, cond),
                "Grouped MRR": t,
                "Split": "Test",
            }
        )
    bar_df = pd.DataFrame(bar_data)

    fig_hl = px.bar(
        bar_df,
        x="Condition",
        y="Grouped MRR",
        color="Split",
        barmode="group",
        text=bar_df["Grouped MRR"].round(3),
        color_discrete_map={"Validation": "#3b82f6", "Test": "#f97316"},
        height=360,
    )
    fig_hl.update_traces(textposition="outside")
    fig_hl.update_layout(
        yaxis=dict(range=[0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=10),
    )
    st.plotly_chart(fig_hl, use_container_width=True)

    # -----------------------------------------------------------------------
    # SECTION B — Per-coverage-pattern
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### B · Coverage Pattern Breakdown (No-Filter)")
    st.markdown(
        "Checks that the pattern-level ordering observed on val is preserved on test. "
        "Test set has ~2x fewer rows per pattern; interpret low-n patterns cautiously."
    )

    df_test_nf = df_test_all[df_test_all["filter_condition"] == "no_filter"].copy()

    cov_order_t = [
        "A+M+S+I",
        "A+M+S",
        "A+M+I",
        "A+M",
        "A+S+I",
        "A+S",
        "A+I",
        "A",
        "M+S+I",
        "M+S",
        "M+I",
        "M",
        "S+I",
        "S",
        "I",
        "NONE",
    ]

    test_cov = (
        df_test_nf.groupby("coverage_pattern")["mrr_grouped"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Test MRR", "count": "Test n"})
        .round(3)
    )
    val_cov_t = (
        df_filter_coverage[df_filter_coverage["filter_applied"] == "none"][
            ["coverage_pattern", "mrr_grouped", "n"]
        ]
        .set_index("coverage_pattern")
        .rename(columns={"mrr_grouped": "Val MRR", "n": "Val n"})
        .round(3)
    )
    cov_df = val_cov_t.join(test_cov, how="outer").fillna(float("nan"))
    cov_df["Gap"] = (cov_df["Test MRR"] - cov_df["Val MRR"]).round(3)
    present_t = [p for p in cov_order_t if p in cov_df.index]
    cov_df = cov_df.reindex(present_t)

    plot_pats = [p for p in present_t if cov_df.loc[p, "Test n"] >= 5]

    cov_long = []
    for p in plot_pats:
        cov_long.append(
            {"Pattern": p, "MRR": cov_df.loc[p, "Val MRR"], "Split": "Validation"}
        )
        cov_long.append(
            {"Pattern": p, "MRR": cov_df.loc[p, "Test MRR"], "Split": "Test"}
        )
    cov_long_df = pd.DataFrame(cov_long)

    fig_cov_t = px.bar(
        cov_long_df,
        x="Pattern",
        y="MRR",
        color="Split",
        barmode="group",
        color_discrete_map={"Validation": "#3b82f6", "Test": "#f97316"},
        category_orders={"Pattern": plot_pats},
        height=380,
        labels={"MRR": "Mean Grouped MRR"},
    )
    fig_cov_t.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8")
    fig_cov_t.update_layout(
        yaxis=dict(range=[0, 1.05]),
        xaxis_tickangle=35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=80),
    )
    st.plotly_chart(fig_cov_t, use_container_width=True)
    st.caption(
        "Patterns with fewer than 5 test rows omitted from chart. Full table below."
    )

    st.markdown("**Full per-pattern table**")
    st.dataframe(
        cov_df.style.format(
            {
                "Val MRR": "{:.3f}",
                "Test MRR": "{:.3f}",
                "Val n": "{:.0f}",
                "Test n": "{:.0f}",
                "Gap": "{:+.3f}",
            }
        ).applymap(
            lambda v: (
                "color: #16a34a"
                if isinstance(v, float) and v > 0.02
                else ("color: #dc2626" if isinstance(v, float) and v < -0.02 else "")
            ),
            subset=["Gap"],
        ),
        use_container_width=True,
    )

    # -----------------------------------------------------------------------
    # SECTION C — Noise robustness
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### C · Noise Robustness")
    st.markdown(
        "Confirms that the noise level ordering (low > medium > high) is preserved on the test set "
        "and that TF-IDF's robustness to omission and compression noise generalizes."
    )

    noise_dims_t = [
        ("noise_level", "Noise Level", ["low", "medium", "high"]),
        ("noise_omission", "Omission Count", None),
        ("noise_compression", "Compression Count", None),
    ]

    col_t1, col_t2, col_t3 = st.columns(3)
    for col_st, (dim, xlabel, order) in zip([col_t1, col_t2, col_t3], noise_dims_t):
        g_val_n = df_filter_noise[dim][
            df_filter_noise[dim]["filter_applied"] == "none"
        ].set_index(dim)["mrr_grouped"]
        g_test_n = df_test_nf.groupby(dim)["mrr_grouped"].mean()
        idx = order if order else sorted(set(g_val_n.index) | set(g_test_n.index))
        idx = [v for v in idx if v in g_val_n.index or v in g_test_n.index]

        noise_long_t = []
        for i in idx:
            noise_long_t.append(
                {
                    "x": str(i),
                    "MRR": g_val_n.get(i, float("nan")),
                    "Split": "Validation",
                }
            )
            noise_long_t.append(
                {"x": str(i), "MRR": g_test_n.get(i, float("nan")), "Split": "Test"}
            )
        noise_df_t = pd.DataFrame(noise_long_t)

        fig_n = px.bar(
            noise_df_t,
            x="x",
            y="MRR",
            color="Split",
            barmode="group",
            color_discrete_map={"Validation": "#3b82f6", "Test": "#f97316"},
            category_orders={"x": [str(i) for i in idx]},
            labels={"x": xlabel, "MRR": "Mean Grouped MRR"},
            height=300,
        )
        fig_n.update_layout(
            yaxis=dict(range=[0, 1.0]),
            showlegend=(dim == "noise_level"),
            margin=dict(l=0, r=0, t=30, b=40),
            title_text=xlabel,
        )
        with col_st:
            st.plotly_chart(fig_n, use_container_width=True)

    st.markdown("**Noise level counts**")
    st.caption(
        "Note that some of the test set counts are low and require careful interpretation."
    )
    df_full_nf = pd.concat(
        [df_elr.assign(split="val"), df_test_nf.assign(split="test")]
    )
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.dataframe(
            df_full_nf.groupby(["split", "noise_level"])
            .size()
            .unstack("split")
            .reset_index(),
            hide_index=True,
        )
    with col_t2:
        st.dataframe(
            df_full_nf.groupby(["split", "noise_omission"])
            .size()
            .unstack("split")
            .reset_index(),
            hide_index=True,
        )
    with col_t3:
        st.dataframe(
            df_full_nf.groupby(["split", "noise_compression"])
            .size()
            .unstack("split")
            .reset_index(),
            hide_index=True,
        )

    # -----------------------------------------------------------------------
    # SECTION D — Per-LOINC-code stability scatter
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### D · Per-LOINC-Code Stability")
    st.markdown(
        "Test set has only a few variants per LOINC code, it is important to interpret per-code numbers as directional "
        "signal only. The scatter confirms that codes that are hard in the val split are also hard in test split "
        "(no systematic overfitting to val-specific surface forms)."
    )

    per_loinc_test = (
        df_test_nf.groupby("true_loinc")["mrr_grouped"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Test MRR", "count": "Test n"})
        .round(3)
    )
    df_filter_loinc = load_filter_loinc()
    per_loinc_val_t = (
        df_filter_loinc[df_filter_loinc["filter_applied"] == "none"]
        .set_index("true_loinc")[["mrr_grouped", "n"]]
        .rename(columns={"mrr_grouped": "Val MRR", "n": "Val n"})
    ).round(3)
    per_loinc_t = per_loinc_val_t.join(per_loinc_test).dropna()
    per_loinc_t["Gap"] = (per_loinc_t["Test MRR"] - per_loinc_t["Val MRR"]).round(3)

    fig_scatter = px.scatter(
        per_loinc_t.reset_index(),
        x="Val MRR",
        y="Test MRR",
        color="Gap",
        size="Test n",
        hover_data=["true_loinc", "Val n", "Test n", "Gap"],
        color_continuous_scale="RdYlGn",
        range_color=[-0.3, 0.3],
        labels={"true_loinc": "LOINC Code"},
        height=460,
    )
    fig_scatter.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="#94a3b8", dash="dash", width=1),
    )
    fig_scatter.update_layout(
        xaxis=dict(range=[0, 1.05], title="Val MRR (per LOINC code)"),
        yaxis=dict(range=[0, 1.05], title="Test MRR (per LOINC code)"),
        margin=dict(l=0, r=0, t=20, b=10),
        coloraxis_colorbar=dict(title="Test−Val"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "Points on the diagonal = stable. Above = test performance better than val. Below = test performance worse than val.  \n"
        "Dot size scales with test n.  \n"
        "Colour: green = test $>$ val, red = test $<$ val."
    )

    col_w, col_b = st.columns(2)
    with col_w:
        st.markdown("**Largest test drops**")
        worst = per_loinc_t.sort_values("Gap").head(5)
        st.dataframe(
            worst[["Val MRR", "Test MRR", "Gap", "Test n"]].style.format(
                {
                    "Val MRR": "{:.3f}",
                    "Test MRR": "{:.3f}",
                    "Gap": "{:+.3f}",
                    "Test n": "{:.0f}",
                }
            ),
            use_container_width=True,
        )
    with col_b:
        st.markdown("**Largest test improvements**")
        best_delta = per_loinc_t.sort_values("Gap", ascending=False).head(5)
        st.dataframe(
            best_delta[["Val MRR", "Test MRR", "Gap", "Test n"]].style.format(
                {
                    "Val MRR": "{:.3f}",
                    "Test MRR": "{:.3f}",
                    "Gap": "{:+.3f}",
                    "Test n": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

    st.caption(
        "⚠️ Per-code test n is small (typically 2–4 variants). Large individual gaps reflect "
        "sampling variance, not systematic overfitting. The aggregate headline MRR is the stable number."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<p style='color:#94a3b8; font-size:0.8rem;'>"
    "LOINC Crosswalk Portfolio Project · TF-IDF Retrieval Benchmark · COVID-19 SARS-CoV-2 ELR Mapping"
    "</p>",
    unsafe_allow_html=True,
)
