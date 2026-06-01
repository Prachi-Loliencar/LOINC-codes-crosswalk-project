# Visualization module for:
#   1. Simulated ELR data quality (noise distributions, structural composition)
#   2. Corpus geometry for TF-IDF and sentence-transformer models
#      (UMAP projections, IDF profiles, inter-code similarity heatmaps)
#
# Design notes:
#   - UMAP is used instead of MDS. MDS on cosine-distance matrices of
#     high-dimensional TF-IDF vectors produces the horseshoe artifact because
#     the distance matrix violates Euclidean assumptions. UMAP operates in the
#     original metric space directly (metric='cosine') and avoids this.
#   - The meaningful corpus comparison is expanded LOINC corpus text vs. ELR
#     strings, since both sides share a surface-form vocabulary after LCN
#     expansion and ELR normalization. Raw LOINC metadata (e.g. "Probe.amp.tar")
#     is not compared to ELR because the vocabulary gap is by construction.

import logging
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.clinical_utils import clean_text
from src.model_building_utils import (
    build_corpus,
    expand_loinc_lcn,
    normalize_elr,
    LOINC_METHOD_TOKENS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette & style defaults
# ---------------------------------------------------------------------------

_PALETTE_METHOD = {
    "naat": "#3498db",
    "antigen": "#e74c3c",
    "mixed/panel": "#9b59b6",
    "unknown": "#95a5a6",
}

_PALETTE_SPECIMEN = {
    "NP": "#2ecc71",
    "NASAL": "#27ae60",
    "THROAT": "#f39c12",
    "SALIVA": "#e67e22",
    "BAL": "#8e44ad",
    "SPUTUM": "#2980b9",
    "COMBINED_NT": "#c0392b",
    "URT_GENERAL": "#1abc9c",
    "LRT_GENERAL": "#d35400",
    "UNKNOWN": "#bdc3c7",
}


# ---------------------------------------------------------------------------
# 1. Simulation quality: noise & structural composition
# ---------------------------------------------------------------------------


def plot_simulation_noise_audit(df_elr: pd.DataFrame) -> plt.Figure:
    """
    Three-panel figure: noise count distribution, noise level breakdown,
    and per-type noise intensity (corruption / compression / omission).
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Simulation Noise Audit", fontsize=15, fontweight="bold", y=1.02)

    # A. Total noise count histogram
    sns.histplot(df_elr["noise_total"], discrete=True, color="#3498db", ax=axes[0])
    axes[0].set_title("Total Noise Count per ELR String")
    axes[0].set_xlabel("Tokens/chars altered")
    axes[0].set_ylabel("Count")

    # B. Noise level categories
    sns.countplot(
        data=df_elr,
        x="noise_level",
        hue="noise_level",
        order=["low", "medium", "high"],
        palette={"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"},
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Noise Level Categorization")
    axes[1].set_xlabel("")

    # C. Per-type noise intensity
    noise_cols = ["noise_corruption", "noise_compression", "noise_omission"]
    noise_melt = df_elr[noise_cols].melt(var_name="type", value_name="count")
    noise_melt["type"] = noise_melt["type"].str.replace("noise_", "").str.title()
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
    axes[2].set_title("Intensity by Noise Type")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Count per string")

    plt.tight_layout()
    return fig


def plot_simulation_structure(df_elr: pd.DataFrame) -> plt.Figure:
    """
    Two-panel figure: component prevalence rates and coverage pattern distribution.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("ELR Structural Composition", fontsize=15, fontweight="bold", y=1.02)

    # A. Component prevalence
    comp_cols = ["has_analyte", "has_method", "has_specimen", "has_model"]
    comp_labels = ["Analyte", "Method", "Specimen", "Model"]
    comp_means = df_elr[comp_cols].mean()
    bars = axes[0].bar(
        comp_labels,
        comp_means.values,
        color=["#3498db", "#e67e22", "#2ecc71", "#9b59b6"],
        edgecolor="white",
        alpha=0.9,
    )
    for bar, val in zip(bars, comp_means.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title("Proportion of ELR Strings Containing Component")
    axes[0].set_ylabel("Prevalence")

    # B. Coverage pattern frequency (top 12)
    top_patterns = df_elr["coverage_pattern"].value_counts().head(12)
    axes[1].barh(
        top_patterns.index[::-1],
        top_patterns.values[::-1],
        color="#3498db",
        alpha=0.85,
        edgecolor="white",
    )
    axes[1].set_title("Top Coverage Patterns")
    axes[1].set_xlabel("Count")

    plt.tight_layout()
    return fig


def plot_noise_by_coverage(df_elr: pd.DataFrame) -> plt.Figure:
    """
    Heatmap: mean noise_total by (coverage_pattern × noise_level).
    Useful for confirming that low-information strings don't just happen to be
    low-noise — the two dimensions are orthogonal by design.
    """
    pivot = df_elr.pivot_table(
        index="coverage_pattern",
        columns="noise_level",
        values="noise_total",
        aggfunc="mean",
    ).reindex(columns=["low", "medium", "high"])
    # Sort rows by total count so dominant patterns are at top
    order = df_elr["coverage_pattern"].value_counts().index
    pivot = pivot.reindex([p for p in order if p in pivot.index])

    fig, ax = plt.subplots(figsize=(7, max(3, len(pivot) * 0.45)))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.4,
        cbar_kws={"label": "Mean total noise"},
    )
    ax.set_title("Mean Noise by Coverage Pattern × Noise Level", fontsize=12)
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Coverage Pattern")
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Token frequency: expanded corpus vs ELR strings
# ---------------------------------------------------------------------------


def _get_token_freq(series: pd.Series, top_n: int = 30) -> pd.DataFrame:
    """Token frequency table from a string Series."""
    tokens = " ".join(series.fillna("").map(clean_text)).split()
    return pd.DataFrame(Counter(tokens).most_common(top_n), columns=["token", "count"])


def plot_corpus_vs_elr_token_freq(
    df_loinc: pd.DataFrame,
    df_elr: pd.DataFrame,
    corpus_strategy: str = "lcn_method_dict_combined",
    rn_stopwords: set = set(),
    top_n: int = 30,
) -> plt.Figure:
    """
    Side-by-side horizontal bar charts comparing token frequencies in:
      - left:  expanded LOINC corpus text (the TF-IDF index side)
      - right: normalized ELR strings (the query side)

    Both sides have had clean_text and strategy-specific expansion applied, so
    the vocabulary gap between raw LOINC metadata and ELR surface forms is
    already bridged. Divergence here reflects genuine coverage gaps.

    Parameters
    ----------
    df_loinc         : covid_surveillance_loinc.csv with 'expanded_lcn' column added
    df_elr           : elr_simulated.csv with 'elr_name_normalized' column added
    corpus_strategy  : corpus strategy to use for the LOINC side
    rn_stopwords     : stopword set from compute_relatednames_stopwords()
    top_n            : number of tokens to show per side
    """
    corpus_series = build_corpus(df_loinc, corpus_strategy, rn_stopwords)
    corpus_freq = _get_token_freq(corpus_series, top_n)
    elr_freq = _get_token_freq(df_elr["elr_name_normalized"], top_n)

    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, top_n * 0.32)))
    fig.suptitle(
        f"Token Frequency: LOINC Corpus ({corpus_strategy}) vs ELR Strings",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    axes[0].barh(
        corpus_freq["token"][::-1],
        corpus_freq["count"][::-1],
        color="#2980b9",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].set_title("LOINC Corpus (expanded, index side)")
    axes[0].set_xlabel("Token count")

    axes[1].barh(
        elr_freq["token"][::-1],
        elr_freq["count"][::-1],
        color="#c0392b",
        alpha=0.85,
        edgecolor="white",
    )
    axes[1].set_title("ELR Strings (normalized, query side)")
    axes[1].set_xlabel("Token count")

    plt.tight_layout()
    return fig


def plot_strategy_idf_profiles(
    df_loinc: pd.DataFrame,
    strategies: list,
    rn_stopwords: set = set(),
    top_n: int = 25,
) -> plt.Figure:
    """
    For each corpus strategy, show the top-N tokens by IDF weight.
    High IDF = rare across documents = discriminative for retrieval.
    Useful for understanding what signal each strategy amplifies.

    Parameters
    ----------
    df_loinc    : LOINC reference with 'expanded_lcn' column
    strategies  : list of corpus strategy names to compare
    rn_stopwords: stopword set
    top_n       : tokens to show per strategy
    """
    n = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, max(6, top_n * 0.3)), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Top Tokens by IDF Weight per Corpus Strategy",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for ax, strategy in zip(axes, strategies):
        corpus = build_corpus(df_loinc, strategy, rn_stopwords)
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True)
        vec.fit(corpus)
        vocab = vec.vocabulary_
        idf = vec.idf_
        idf_df = (
            pd.DataFrame(
                {"token": list(vocab.keys()), "idf": idf[list(vocab.values())]}
            )
            .sort_values("idf", ascending=False)
            .head(top_n)
        )
        ax.barh(
            idf_df["token"][::-1],
            idf_df["idf"][::-1],
            color="#8e44ad",
            alpha=0.8,
            edgecolor="white",
        )
        ax.set_title(strategy, fontsize=10)
        ax.set_xlabel("IDF weight")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Corpus geometry: UMAP projections
# ---------------------------------------------------------------------------


def _build_umap_embedding(matrix, n_neighbors: int = 12, min_dist: float = 0.1):
    """
    Fit UMAP on a sparse or dense matrix using cosine metric.
    Returns (embedding_2d, reducer).
    UMAP with metric='cosine' operates in the original high-dimensional space,
    avoiding the Euclidean-assumption violation that causes MDS horseshoe artifacts.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    return reducer.fit_transform(matrix), reducer


def plot_tfidf_corpus_umap(
    df_loinc: pd.DataFrame,
    strategies: list,
    rn_stopwords: set = set(),
    ngram_range: tuple = (1, 1),
    color_by: str = "method_typ",
    n_neighbors: int = 12,
) -> plt.Figure:
    """
    One UMAP panel per corpus strategy, points colored by LOINC metadata axis.
    Each point is one LOINC code. Tight clusters indicate that the corpus
    representation groups similar codes together — the property that makes
    retrieval robust.

    Parameters
    ----------
    df_loinc     : LOINC reference with 'expanded_lcn' column
    strategies   : list of corpus strategy names
    rn_stopwords : stopword set
    ngram_range  : word ngram range for TF-IDF (e.g. (1,1) or (1,2))
    color_by     : 'method_typ', 'system', or 'method_class'
    n_neighbors  : UMAP n_neighbors — smaller values = finer local structure
    """
    n = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"LOINC Corpus UMAP — TF-IDF word {ngram_range}, colored by {color_by}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    labels = df_loinc[color_by].fillna("unknown").astype(str)
    unique_labels = labels.unique()
    palette = sns.color_palette("tab10", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    for ax, strategy in zip(axes, strategies):
        corpus = build_corpus(df_loinc, strategy, rn_stopwords)
        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=1,
            max_df=0.85,
        )
        matrix = vec.fit_transform(corpus)
        emb, _ = _build_umap_embedding(matrix, n_neighbors=n_neighbors)

        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                label=label,
                color=color_map[label],
                s=80,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.4,
            )

        ax.set_title(strategy, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Single shared legend on the rightmost axis
    axes[-1].legend(
        title=color_by,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        framealpha=0.8,
    )
    plt.tight_layout()
    return fig


def plot_tfidf_joint_umap(
    df_loinc: pd.DataFrame,
    df_elr: pd.DataFrame,
    corpus_strategy: str = "lcn_method_dict_combined",
    rn_stopwords: set = set(),
    ngram_range: tuple = (1, 1),
    n_neighbors: int = 15,
    elr_sample: int = 300,
) -> plt.Figure:
    """
    Jointly embeds LOINC corpus documents and ELR query strings in the same
    UMAP space (fit on corpus, transform queries). Points from the two
    populations are rendered separately so you can see whether ELR queries
    land near their true LOINC code.

    LOINC points are larger and labeled; ELR points are small and colored by
    their true loinc_num. Good retrieval → ELR points cluster around the
    matching LOINC point.

    Parameters
    ----------
    df_loinc        : LOINC reference with 'expanded_lcn' column
    df_elr          : ELR dataset with 'elr_name_normalized' and 'loinc_num'
    corpus_strategy : corpus strategy for the LOINC side
    rn_stopwords    : stopword set
    ngram_range     : word TF-IDF ngram range
    n_neighbors     : UMAP n_neighbors
    elr_sample      : number of ELR strings to plot (random sample)
    """
    corpus = build_corpus(df_loinc, corpus_strategy, rn_stopwords)
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1,
        max_df=0.85,
    )
    loinc_matrix = vec.fit_transform(corpus)

    # Sample ELR strings
    df_elr_sample = df_elr.sample(min(elr_sample, len(df_elr)), random_state=42).copy()
    elr_matrix = vec.transform(df_elr_sample["elr_name_normalized"].fillna(""))

    import umap

    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
    )
    loinc_emb = reducer.fit_transform(loinc_matrix)
    elr_emb = reducer.transform(elr_matrix)

    unique_loincs = df_loinc["loinc_num"].values
    palette = sns.color_palette("tab20", len(unique_loincs))
    loinc_color = dict(zip(unique_loincs, palette))

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(
        f"Joint UMAP: LOINC Corpus vs ELR Queries\n({corpus_strategy}, "
        f"TF-IDF word {ngram_range})",
        fontsize=13,
        fontweight="bold",
    )

    # ELR points (small, colored by true LOINC)
    for loinc_num in unique_loincs:
        mask = df_elr_sample["loinc_num"] == loinc_num
        if mask.any():
            ax.scatter(
                elr_emb[mask.values, 0],
                elr_emb[mask.values, 1],
                color=loinc_color[loinc_num],
                s=18,
                alpha=0.55,
                marker="o",
                edgecolors="none",
            )

    # LOINC corpus points (large, labeled)
    for i, (loinc_num, lcn) in enumerate(
        zip(df_loinc["loinc_num"].values, df_loinc["long_common_name"].values)
    ):
        ax.scatter(
            loinc_emb[i, 0],
            loinc_emb[i, 1],
            color=loinc_color[loinc_num],
            s=200,
            alpha=1.0,
            marker="*",
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            loinc_num,
            (loinc_emb[i, 0], loinc_emb[i, 1]),
            fontsize=6,
            ha="left",
            va="bottom",
            xytext=(3, 3),
            textcoords="offset points",
            color="#2c3e50",
        )

    ax.set_xticks([])
    ax.set_yticks([])

    # Legend: circle=ELR, star=LOINC corpus
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#777",
            markersize=6,
            label="ELR query (colored by true LOINC)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#777",
            markersize=10,
            label="LOINC corpus document",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.8)
    plt.tight_layout()
    return fig


def plot_st_corpus_umap(
    df_loinc: pd.DataFrame,
    df_elr: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    color_by: str = "method_typ",
    n_neighbors: int = 12,
    elr_sample: int = 300,
) -> plt.Figure:
    """
    Sentence-transformer version of the joint UMAP.
    Encodes LOINC long_common_name (no expansion needed — ST handles paraphrase)
    and ELR strings, projects both into 2D.

    Parameters
    ----------
    df_loinc    : LOINC reference table (needs long_common_name, loinc_num, color_by col)
    df_elr      : ELR dataset (needs elr_name, loinc_num)
    model_name  : HuggingFace sentence-transformer model name
    color_by    : column on df_loinc to use for LOINC point colors
    n_neighbors : UMAP n_neighbors
    elr_sample  : number of ELR strings to embed (can be slow for large datasets)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required: pip install sentence-transformers"
        )

    import umap

    import logging as _logging

    _st_logger = _logging.getLogger("sentence_transformers")
    _prev_level = _st_logger.level
    _st_logger.setLevel(_logging.ERROR)
    model = SentenceTransformer(model_name)
    _st_logger.setLevel(_prev_level)

    loinc_texts = df_loinc["long_common_name"].fillna("").tolist()
    loinc_emb = model.encode(
        loinc_texts, show_progress_bar=True, normalize_embeddings=True
    )

    df_elr_sample = df_elr.sample(min(elr_sample, len(df_elr)), random_state=42).copy()
    elr_texts = df_elr_sample["elr_name"].fillna("").tolist()
    elr_emb = model.encode(elr_texts, show_progress_bar=True, normalize_embeddings=True)

    all_emb = np.vstack([loinc_emb, elr_emb])
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
    )
    all_2d = reducer.fit_transform(all_emb)
    loinc_2d = all_2d[: len(loinc_emb)]
    elr_2d = all_2d[len(loinc_emb) :]

    labels = df_loinc[color_by].fillna("unknown").astype(str)
    unique_labels = labels.unique()
    palette = sns.color_palette("tab10", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    unique_loincs = df_loinc["loinc_num"].values
    loinc_color_by_code = dict(zip(unique_loincs, [color_map[l] for l in labels]))

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(
        f"Joint UMAP: Sentence Transformer ({model_name.split('/')[-1]})\n"
        f"LOINC corpus vs ELR queries, colored by {color_by}",
        fontsize=12,
        fontweight="bold",
    )

    # ELR points
    for i, (_, row) in enumerate(df_elr_sample.iterrows()):
        c = loinc_color_by_code.get(row["loinc_num"], "#bdc3c7")
        ax.scatter(
            elr_2d[i, 0], elr_2d[i, 1], color=c, s=15, alpha=0.45, edgecolors="none"
        )

    # LOINC corpus points
    for i, (loinc_num, label) in enumerate(zip(df_loinc["loinc_num"].values, labels)):
        ax.scatter(
            loinc_2d[i, 0],
            loinc_2d[i, 1],
            color=color_map[label],
            s=180,
            alpha=1.0,
            marker="*",
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            loinc_num,
            (loinc_2d[i, 0], loinc_2d[i, 1]),
            fontsize=6,
            ha="left",
            va="bottom",
            xytext=(3, 3),
            textcoords="offset points",
            color="#2c3e50",
        )

    # Legend for color_by categories
    from matplotlib.lines import Line2D

    cat_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[l],
            markersize=8,
            label=l,
        )
        for l in unique_labels
    ]
    cat_handles += [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#777",
            markersize=10,
            label="LOINC corpus (★)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#777",
            markersize=6,
            label="ELR query (●)",
        ),
    ]
    ax.legend(
        handles=cat_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        framealpha=0.8,
        title=color_by,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Inter-code similarity heatmaps under different corpus strategies
# ---------------------------------------------------------------------------


def plot_loinc_similarity_clustermap(
    df_loinc: pd.DataFrame,
    corpus_strategy: str = "lcn_method_dict_combined",
    rn_stopwords: set = set(),
    ngram_range: tuple = (1, 1),
    eval_loincs: Optional[list] = None,
) -> sns.matrix.ClusterGrid:
    """
    Annotated clustermap of cosine similarity between LOINC codes.

    The corpus contains 98 unique LOINC codes after preprocessing, which is
    too many for readable per-code labels. Two modes are supported:

      eval_loincs=None  : full 98-code corpus, no tick labels — structure is
                          read from the colored sidebars alone. Useful for
                          showing the overall landscape.
      eval_loincs=[...] : restrict to the 13 eval-target codes. Labels are
                          shown and cell values annotated. Useful for the
                          portfolio portrait where interpretability matters.

    Row/column ordering is determined by hierarchical clustering of the
    similarity matrix — no manual sort column needed.
    Colored sidebars show method_typ and system so you can read off whether
    the corpus cleanly separates NAAT from antigen codes.

    Use plot_within_between_similarity for cross-strategy comparison.

    Parameters
    ----------
    df_loinc         : LOINC reference with 'expanded_lcn', 'method_typ',
                       'system', 'long_common_name', 'loinc_num' columns
    corpus_strategy  : single corpus strategy to visualize
    rn_stopwords     : stopword set from compute_relatednames_stopwords()
    ngram_range      : word TF-IDF ngram range
    eval_loincs      : optional list of loinc_num strings to restrict display
                       to eval-target codes (e.g. the 13 codes with seeds)
    """
    # Always fit the vectorizer on the full corpus so IDF weights are realistic
    corpus_full = build_corpus(df_loinc, corpus_strategy, rn_stopwords)
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1,
        max_df=0.85,
    )
    matrix_full = normalize(vec.fit_transform(corpus_full))

    if eval_loincs is not None:
        # Restrict display to eval-target rows, keeping IDF from full corpus
        mask = df_loinc["loinc_num"].isin(eval_loincs)
        df_display = df_loinc[mask].copy()
        matrix_display = matrix_full[mask.values]
        show_labels = True
        annot = True
        figsize = (13, 13)
        linewidths = 0.5
    else:
        df_display = df_loinc
        matrix_display = matrix_full
        show_labels = False
        annot = False
        figsize = (14, 14)
        linewidths = 0.0

    sim = pd.DataFrame(
        cosine_similarity(matrix_display),
        index=df_display["loinc_num"].values,
        columns=df_display["loinc_num"].values,
    )

    # --- Sidebar color palettes ---
    method_vals = df_display["method_typ"].fillna("unknown").astype(str)
    unique_methods = sorted(method_vals.unique())
    method_palette = dict(
        zip(unique_methods, sns.color_palette("Set2", len(unique_methods)))
    )
    method_colors = pd.Series(
        method_vals.map(method_palette).values, index=df_display["loinc_num"].values
    )

    system_vals = df_display["system"].fillna("unknown").astype(str)
    unique_systems = sorted(system_vals.unique())
    system_palette = dict(
        zip(unique_systems, sns.color_palette("Set1", len(unique_systems)))
    )
    system_colors = pd.Series(
        system_vals.map(system_palette).values, index=df_display["loinc_num"].values
    )

    row_colors = pd.DataFrame(
        {"method_typ": method_colors, "system": system_colors},
        index=df_display["loinc_num"].values,
    )

    cg = sns.clustermap(
        sim,
        cmap="viridis",
        vmin=0,
        vmax=1,
        row_colors=row_colors,
        col_colors=row_colors,
        figsize=figsize,
        linewidths=linewidths,
        annot=annot,
        fmt=".2f" if annot else "",
        annot_kws={"size": 7} if annot else {},
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        dendrogram_ratio=0.12,
        xticklabels=show_labels,
        yticklabels=show_labels,
    )
    if show_labels:
        cg.ax_heatmap.set_xticklabels(
            cg.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=8
        )
        cg.ax_heatmap.set_yticklabels(
            cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8
        )

    # Sidebar legends — one combined legend on the figure with a visual separator
    # between the two annotation axes (method_typ and system).
    # ax_col_dendrogram is used as an invisible anchor; the legend is placed
    # via figure coordinates so it doesn't compete with the dendrogram content.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = []
    legend_handles.append(
        Patch(facecolor="none", edgecolor="none", label="── method_typ ──")
    )
    for label, color in method_palette.items():
        legend_handles.append(Patch(facecolor=color, edgecolor="white", label=label))

    legend_handles.append(
        Patch(facecolor="none", edgecolor="none", label="── system ──")
    )
    for label, color in system_palette.items():
        legend_handles.append(Patch(facecolor=color, edgecolor="white", label=label))

    cg.figure.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.72),
        fontsize=7,
        framealpha=0.8,
        handlelength=1.2,
        handleheight=0.9,
    )

    mode_label = (
        f"eval codes only (n={len(df_display)})"
        if eval_loincs
        else f"full corpus (n={len(df_display)})"
    )
    cg.figure.suptitle(
        f"LOINC Code Cosine Similarity — {corpus_strategy}\n{mode_label}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    return cg


def plot_within_between_similarity(
    df_loinc: pd.DataFrame,
    strategies: list,
    rn_stopwords: set = set(),
    ngram_range: tuple = (1, 1),
    group_by: str = "method_typ",
    effect_size: bool = False,
) -> plt.Figure:
    """
    For each corpus strategy: computes mean within-group and between-group
    cosine similarity, then plots them alongside a separation metric.

    Operates on unique LOINC codes (diagonal excluded). Two modes:

      effect_size=False (default): raw means + ratio (within / between).
        Useful for understanding absolute similarity levels within a single
        method. The ratio cancels the scale so strategies are comparable
        to each other, but TF-IDF and ST ratios are still not comparable
        because the geometry of the two spaces differs.

      effect_size=True: raw means + Cohen's d.
        d = (mean_within - mean_between) / pooled_std
        Divides by the pooled standard deviation of all pairwise similarities,
        removing both the mean-level compression effect and the scale
        difference between methods. Cohen's d is directly comparable across
        TF-IDF and ST and is the correct metric when comparing the two.
        Conventional benchmarks: d ≈ 0.2 small, 0.5 medium, 0.8 large.

    Parameters
    ----------
    df_loinc     : LOINC reference with 'expanded_lcn' column
    strategies   : list of corpus strategy names to compare
    rn_stopwords : stopword set
    ngram_range  : word TF-IDF ngram range
    group_by     : LOINC axis to define groups — 'method_typ' or 'system'
    effect_size  : if True, report Cohen's d instead of within/between ratio
    """
    df_unique = df_loinc.groupby("loinc_num").first().reset_index()

    records = []
    for strategy in strategies:
        corpus = build_corpus(df_unique, strategy, rn_stopwords)
        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
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
                within.append(sim[i, j]) if groups[i] == groups[j] else between.append(
                    sim[i, j]
                )

        w_arr = np.array(within)
        b_arr = np.array(between)
        mean_within = w_arr.mean()
        mean_between = b_arr.mean()
        pooled_std = np.sqrt((w_arr.std(ddof=1) ** 2 + b_arr.std(ddof=1) ** 2) / 2)
        cohens_d = (mean_within - mean_between) / pooled_std if pooled_std > 0 else 0.0
        ratio = mean_within / mean_between if mean_between > 0 else float("inf")

        records.append(
            {
                "strategy": strategy,
                "within": mean_within,
                "between": mean_between,
                "separation": cohens_d if effect_size else ratio,
            }
        )

    df_rec = pd.DataFrame(records).sort_values("separation", ascending=False)

    es_note = " — Cohen's d" if effect_size else ""
    sep_label = (
        "Cohen's d (pooled-std normalized)"
        if effect_size
        else "Ratio (within / between)"
    )
    sep_fmt = lambda v: f"{v:.3f}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Within- vs Between-Group Similarity by Corpus Strategy{es_note}\n(grouped by {group_by})",
        fontsize=13,
        fontweight="bold",
    )

    x = np.arange(len(df_rec))
    w = 0.35
    axes[0].bar(
        x - w / 2,
        df_rec["within"],
        width=w,
        label="Within group",
        color="#2980b9",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].bar(
        x + w / 2,
        df_rec["between"],
        width=w,
        label="Between group",
        color="#e74c3c",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_rec["strategy"], rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean cosine similarity (raw)")
    axes[0].set_title("Within vs Between Group Similarity")
    axes[0].legend(fontsize=9)
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_axisbelow(True)

    bars = axes[1].bar(
        df_rec["strategy"],
        df_rec["separation"],
        color="#8e44ad",
        alpha=0.85,
        edgecolor="white",
    )
    for bar, val in zip(bars, df_rec["separation"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            sep_fmt(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if effect_size:
        for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
            axes[1].axhline(
                thresh, color="grey", linewidth=0.8, linestyle=":", alpha=0.7
            )
            axes[1].text(
                len(df_rec) - 0.5,
                thresh + 0.01,
                lbl,
                fontsize=7,
                color="grey",
                ha="right",
            )
    axes[1].set_xticks(range(len(df_rec)))
    axes[1].set_xticklabels(df_rec["strategy"], rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel(sep_label)
    axes[1].set_title(f"{sep_label} — higher = better discriminability")
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.4)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_st_within_between_similarity(
    df_loinc: pd.DataFrame,
    model_names: list,
    group_by: str = "method_typ",
    effect_size: bool = False,
) -> plt.Figure:
    """
    Sentence-transformer counterpart of plot_within_between_similarity.

    For each ST model: encodes LOINC long_common_name strings, computes
    pairwise cosine similarity between unique LOINC codes (diagonal excluded),
    then reports mean within/between similarity and a separation metric.

    Two modes — same semantics as plot_within_between_similarity:
      effect_size=False : raw means + ratio (within / between)
      effect_size=True  : raw means + Cohen's d, directly comparable to
                          the TF-IDF equivalent

    The ablation axis is model choice. Candidate models:
      - 'sentence-transformers/all-MiniLM-L6-v2'   (fast, general)
      - 'sentence-transformers/all-mpnet-base-v2'  (stronger, general)
      - 'pritamdeka/S-PubMedBert-MS-MARCO'         (biomedical)
      - 'NLP4Science/BioLORD-2023'                 (clinical ontology)

    Parameters
    ----------
    df_loinc     : LOINC reference with 'long_common_name', 'loinc_num',
                   and group_by column
    model_names  : list of HuggingFace model name strings
    group_by     : LOINC axis to define groups — 'method_typ' or 'system'
    effect_size  : if True, report Cohen's d instead of within/between ratio
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required: pip install sentence-transformers"
        )

    df_unique = df_loinc.groupby("loinc_num").first().reset_index()
    groups = df_unique[group_by].fillna("unknown").astype(str).values
    texts = df_unique["long_common_name"].fillna("").tolist()
    n = len(groups)

    records = []
    for model_name in model_names:
        import logging as _logging

        _st_logger = _logging.getLogger("sentence_transformers")
        _prev_level = _st_logger.level
        _st_logger.setLevel(_logging.ERROR)
        model = SentenceTransformer(model_name)
        _st_logger.setLevel(_prev_level)
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sim = cosine_similarity(emb)

        within, between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                within.append(sim[i, j]) if groups[i] == groups[j] else between.append(
                    sim[i, j]
                )

        w_arr = np.array(within)
        b_arr = np.array(between)
        mean_within = w_arr.mean()
        mean_between = b_arr.mean()
        pooled_std = np.sqrt((w_arr.std(ddof=1) ** 2 + b_arr.std(ddof=1) ** 2) / 2)
        cohens_d = (mean_within - mean_between) / pooled_std if pooled_std > 0 else 0.0
        ratio = mean_within / mean_between if mean_between > 0 else float("inf")

        records.append(
            {
                "model": model_name.split("/")[-1],
                "within": mean_within,
                "between": mean_between,
                "separation": cohens_d if effect_size else ratio,
            }
        )

    df_rec = pd.DataFrame(records).sort_values("separation", ascending=False)

    es_note = " — Cohen's d" if effect_size else ""
    sep_label = (
        "Cohen's d (pooled-std normalized)"
        if effect_size
        else "Ratio (within / between)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Within- vs Between-Group Similarity by ST Model{es_note}\n(grouped by {group_by})",
        fontsize=13,
        fontweight="bold",
    )

    x = np.arange(len(df_rec))
    w = 0.35
    axes[0].bar(
        x - w / 2,
        df_rec["within"],
        width=w,
        label="Within group",
        color="#2980b9",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].bar(
        x + w / 2,
        df_rec["between"],
        width=w,
        label="Between group",
        color="#e74c3c",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_rec["model"], rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean cosine similarity (raw)")
    axes[0].set_title("Within vs Between Group Similarity")
    axes[0].legend(fontsize=9)
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_axisbelow(True)

    bars = axes[1].bar(
        df_rec["model"],
        df_rec["separation"],
        color="#8e44ad",
        alpha=0.85,
        edgecolor="white",
    )
    for bar, val in zip(bars, df_rec["separation"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if effect_size:
        for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
            axes[1].axhline(
                thresh, color="grey", linewidth=0.8, linestyle=":", alpha=0.7
            )
            axes[1].text(
                len(df_rec) - 0.5,
                thresh + 0.01,
                lbl,
                fontsize=7,
                color="grey",
                ha="right",
            )
    axes[1].set_xticks(range(len(df_rec)))
    axes[1].set_xticklabels(df_rec["model"], rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel(sep_label)
    axes[1].set_title(f"{sep_label} — higher = better discriminability")
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.4)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_similarity_distributions(
    tfidf_results: dict,
    st_results: dict,
    group_by: str = "method_typ",
) -> plt.Figure:
    """
    Visualizes the full distribution of pairwise cosine similarities, split
    into within-group and between-group pairs, for one TF-IDF configuration
    and one ST model side by side.

    This is the diagnostic complement to the within/between bar charts:
    the bar charts show means, this shows the full shape. Overlapping
    within/between distributions indicate poor discriminability regardless
    of what the means suggest. Rightward shift of the ST panel (both
    distributions shifted right relative to TF-IDF) indicates compression.

    Cohen's d is annotated on each panel so the effect size is readable
    directly from the distribution plot without cross-referencing the bar
    chart. The raw x-axis is kept (no centering) so compression is visible
    as the absolute position of the distributions.

    Each result dict must be pre-computed by _compute_sim_pairs() so
    encoding/vectorization happens once and this plot can be called cheaply.

    Parameters
    ----------
    tfidf_results : dict with keys 'label', 'within', 'between'
                    from _compute_sim_pairs() for a TF-IDF config
    st_results    : dict with keys 'label', 'within', 'between'
                    from _compute_sim_pairs() for an ST model
    group_by      : axis used to define groups (for subtitle only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Pairwise Cosine Similarity Distributions\n"
        f"Within- vs Between-Group ({group_by})",
        fontsize=13,
        fontweight="bold",
    )

    for ax, result in zip(axes, [tfidf_results, st_results]):
        w = np.array(result["within"])
        b = np.array(result["between"])

        pooled_std = np.sqrt((w.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        d = (w.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

        # KDE + rug for within
        sns.kdeplot(
            w,
            ax=ax,
            color="#2980b9",
            linewidth=2,
            fill=True,
            alpha=0.25,
            label="Within group",
        )
        ax.plot(
            w,
            np.zeros_like(w) - 0.02,
            "|",
            color="#2980b9",
            alpha=0.3,
            markersize=4,
            transform=ax.get_xaxis_transform(),
        )

        # KDE + rug for between
        sns.kdeplot(
            b,
            ax=ax,
            color="#e74c3c",
            linewidth=2,
            fill=True,
            alpha=0.25,
            label="Between group",
        )
        ax.plot(
            b,
            np.zeros_like(b) - 0.05,
            "|",
            color="#e74c3c",
            alpha=0.3,
            markersize=4,
            transform=ax.get_xaxis_transform(),
        )

        # Vertical mean lines
        ax.axvline(w.mean(), color="#2980b9", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axvline(b.mean(), color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.8)

        # Cohen's d annotation in upper corner
        ax.text(
            0.97,
            0.95,
            f"Cohen's d = {d:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8
            ),
        )

        ax.set_xlabel("Cosine similarity (raw)")
        ax.set_ylabel("Density")
        ax.set_title(result["label"])
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def _compute_sim_pairs(
    matrix_normalized: np.ndarray,
    groups: np.ndarray,
    label: str,
) -> dict:
    """
    Low-level helper: extracts upper-triangle within/between pair lists from
    a pre-normalized similarity matrix.

    Prefer the high-level wrappers for notebook use:
      _sim_pairs_tfidf() — builds TF-IDF matrix then calls this
      _sim_pairs_st()    — encodes with ST then calls this

    Parameters
    ----------
    matrix_normalized : L2-normalized feature matrix (n_codes × n_features)
    groups            : array of group labels aligned with matrix rows
    label             : display label for this configuration

    Returns
    -------
    dict with keys 'label', 'within', 'between'
    """
    sim = cosine_similarity(matrix_normalized)
    n = len(groups)
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if groups[i] == groups[j] else between).append(sim[i, j])
    return {"label": label, "within": within, "between": between}


def _sim_pairs_tfidf(
    df_loinc: pd.DataFrame,
    corpus_strategy: str,
    rn_stopwords: set,
    group_by: str = "method_typ",
    ngram_range: tuple = (1, 1),
) -> dict:
    """
    Builds a TF-IDF corpus from df_loinc under corpus_strategy, computes
    pairwise cosine similarities between unique LOINC codes, and returns the
    within/between pair lists ready for plot_similarity_distributions.

    Operates on unique LOINC codes (one row per loinc_num) so results are
    not inflated by seed duplication.

    Parameters
    ----------
    df_loinc         : LOINC reference with 'expanded_lcn' and group_by column
    corpus_strategy  : corpus strategy name (passed to build_corpus)
    rn_stopwords     : stopword set from compute_relatednames_stopwords()
    group_by         : LOINC axis defining groups — 'method_typ' or 'system'
    ngram_range      : word TF-IDF ngram range

    Returns
    -------
    dict with keys 'label', 'within', 'between'

    Example
    -------
    >>> tfidf_pairs = _sim_pairs_tfidf(
    ...     loinc, "lcn_method_dict_combined", rn_stopwords
    ... )
    """
    df_unique = df_loinc.groupby("loinc_num").first().reset_index()
    groups = df_unique[group_by].fillna("unknown").astype(str).values
    corpus = build_corpus(df_unique, corpus_strategy, rn_stopwords)
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )
    matrix = normalize(vec.fit_transform(corpus))
    return _compute_sim_pairs(matrix, groups, f"{corpus_strategy} (TF-IDF)")


def _sim_pairs_st(
    df_loinc: pd.DataFrame,
    model_name: str,
    group_by: str = "method_typ",
    input_col: str = "long_common_name",
) -> dict:
    """
    Encodes LOINC texts with a sentence-transformer model, computes pairwise
    cosine similarities between unique LOINC codes, and returns the
    within/between pair lists ready for plot_similarity_distributions.

    Operates on unique LOINC codes (one row per loinc_num).

    Parameters
    ----------
    df_loinc    : LOINC reference with input_col and group_by columns
    model_name  : HuggingFace sentence-transformer model name string
    group_by    : LOINC axis defining groups — 'method_typ' or 'system'
    input_col   : column to encode. Defaults to 'long_common_name' (no
                  LCN expansion — the ST model handles paraphrase natively).
                  Pass 'expanded_lcn' to test whether feeding the expanded
                  corpus text to the ST encoder changes discriminability.

    Returns
    -------
    dict with keys 'label', 'within', 'between'

    Example
    -------
    >>> st_pairs = _sim_pairs_st(
    ...     loinc, "sentence-transformers/all-MiniLM-L6-v2"
    ... )
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required: pip install sentence-transformers"
        )

    df_unique = df_loinc.groupby("loinc_num").first().reset_index()
    groups = df_unique[group_by].fillna("unknown").astype(str).values
    texts = df_unique[input_col].fillna("").tolist()
    import logging as _logging

    _st_logger = _logging.getLogger("sentence_transformers")
    _prev_level = _st_logger.level
    _st_logger.setLevel(_logging.ERROR)
    model = SentenceTransformer(model_name)
    _st_logger.setLevel(_prev_level)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    short = model_name.split("/")[-1]
    return _compute_sim_pairs(emb, groups, f"{short} (ST, {input_col})")


def plot_similarity_distributions_from_names(
    df_loinc: pd.DataFrame,
    tfidf_strategy: str,
    st_model_name: str,
    rn_stopwords: set,
    group_by: str = "method_typ",
    ngram_range: tuple = (1, 1),
    st_input_col: str = "long_common_name",
) -> plt.Figure:
    """
    Notebook-friendly single-call wrapper for plot_similarity_distributions.

    Accepts strategy/model name strings and dataframes directly — no manual
    matrix construction or pair extraction needed. Internally calls
    _sim_pairs_tfidf() and _sim_pairs_st() then passes results to
    plot_similarity_distributions().

    Parameters
    ----------
    df_loinc         : LOINC reference with 'expanded_lcn', 'long_common_name',
                       'loinc_num', and group_by column
    tfidf_strategy   : corpus strategy name for the TF-IDF panel
    st_model_name    : HuggingFace model name string for the ST panel
    rn_stopwords     : stopword set from compute_relatednames_stopwords()
    group_by         : LOINC axis defining groups — 'method_typ' or 'system'
    ngram_range      : word TF-IDF ngram range
    st_input_col     : column to encode for ST ('long_common_name' or
                       'expanded_lcn')

    Returns
    -------
    plt.Figure

    Example
    -------
    >>> fig = plot_similarity_distributions_from_names(
    ...     loinc,
    ...     tfidf_strategy="lcn_method_dict_combined",
    ...     st_model_name="sentence-transformers/all-MiniLM-L6-v2",
    ...     rn_stopwords=rn_stopwords,
    ... )
    """
    tfidf_pairs = _sim_pairs_tfidf(
        df_loinc,
        tfidf_strategy,
        rn_stopwords,
        group_by=group_by,
        ngram_range=ngram_range,
    )
    st_pairs = _sim_pairs_st(
        df_loinc,
        st_model_name,
        group_by=group_by,
        input_col=st_input_col,
    )
    return plot_similarity_distributions(tfidf_pairs, st_pairs, group_by=group_by)


# ---------------------------------------------------------------------------
# 5. ELR string diagnostics
# ---------------------------------------------------------------------------


def plot_elr_length_distribution(df_elr: pd.DataFrame) -> plt.Figure:
    """
    Token count distribution of ELR strings, broken down by noise level and
    coverage pattern. Confirms that noise level and coverage pattern are
    independently distributed — a basic sanity check on the simulation.

    Parameters
    ----------
    df_elr : elr_simulated.csv as a DataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("ELR String Length (Token Count)", fontsize=13, fontweight="bold")

    # A. Distribution by noise level
    sns.boxplot(
        data=df_elr,
        x="noise_level",
        y="analyte_len",
        order=["low", "medium", "high"],
        palette={"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"},
        ax=axes[0],
    )
    axes[0].set_title("Token Count by Noise Level")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Token count")

    # B. Mean token count by coverage pattern
    mean_len = (
        df_elr.groupby("coverage_pattern")["analyte_len"]
        .mean()
        .sort_values(ascending=False)
    )
    axes[1].barh(
        mean_len.index[::-1],
        mean_len.values[::-1],
        color="#3498db",
        alpha=0.85,
        edgecolor="white",
    )
    axes[1].set_title("Mean Token Count by Coverage Pattern")
    axes[1].set_xlabel("Mean token count")

    plt.tight_layout()
    return fig


def plot_elr_token_collision_audit(df_elr: pd.DataFrame) -> None:
    """
    Prints a text report of ELR strings that map to more than one LOINC code
    (ambiguous strings). Also reports unique string count and collision rate.
    These are simulation artifacts worth tracking: a too-high collision rate
    means the perturbation pipeline is generating insufficiently distinctive
    strings.

    Parameters
    ----------
    df_elr : elr_simulated.csv as a DataFrame
    """
    collisions = df_elr.groupby("elr_name")["loinc_num"].nunique()
    n_ambiguous = (collisions > 1).sum()
    n_total = len(df_elr)
    n_unique_strings = collisions.shape[0]

    print(f"Total ELR strings:          {n_total:,}")
    print(f"Unique ELR strings:         {n_unique_strings:,}")
    print(f"Ambiguous (>1 LOINC code):  {n_ambiguous:,}")
    print(f"Collision rate:             {n_ambiguous / n_unique_strings:.2%}")

    if n_ambiguous > 0:
        print("\nTop 5 ambiguous strings:")
        bad = collisions[collisions > 1].sort_values(ascending=False).head(5)
        for elr_str, n_codes in bad.items():
            codes = df_elr.loc[df_elr["elr_name"] == elr_str, "loinc_num"].unique()
            print(f"  '{elr_str}' -> {n_codes} codes: {codes.tolist()}")


# ---------------------------------------------------------------------------
# Example notebook usage (not executed on import)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Load data exactly as ablation.py does — run ablation.load_data()    #
    # first or replicate the steps below.                                 #
    # ------------------------------------------------------------------ #
    from src.model_building_utils import (
        expand_loinc_lcn,
        normalize_elr,
        compute_relatednames_stopwords,
    )
    from src.clinical_utils import clean_text

    elr = pd.read_csv("data/processed/elr_simulated.csv")
    loinc = pd.read_csv("data/processed/covid_surveillance_loinc.csv")
    loinc = loinc[~loinc.method_typ.isna()].copy()
    loinc["expanded_lcn"] = (
        loinc["long_common_name"].map(clean_text).map(expand_loinc_lcn)
    )
    elr["elr_name_normalized"] = elr["elr_name"].map(clean_text).map(normalize_elr)
    rn_stopwords = compute_relatednames_stopwords(loinc, threshold=0.85)

    STRATEGIES_TO_COMPARE = [
        "lcn_only",
        "lcn_method_dict_combined",
        "lcn_method_dict_filtered_rn",
        "component_weighted_method_dict",
    ]

    # 1. Simulation noise audit
    fig1 = plot_simulation_noise_audit(elr)
    fig1.savefig("data/results/viz_noise_audit.png", dpi=150, bbox_inches="tight")

    # 2. Structural composition
    fig2 = plot_simulation_structure(elr)
    fig2.savefig("data/results/viz_structure.png", dpi=150, bbox_inches="tight")

    # 3. Noise × coverage pattern heatmap
    fig3 = plot_noise_by_coverage(elr)
    fig3.savefig("data/results/viz_noise_by_coverage.png", dpi=150, bbox_inches="tight")

    # 4. Corpus vs ELR token frequencies (best strategy)
    fig4 = plot_corpus_vs_elr_token_freq(
        loinc,
        elr,
        corpus_strategy="lcn_method_dict_combined",
        rn_stopwords=rn_stopwords,
    )
    fig4.savefig("data/results/viz_token_freq.png", dpi=150, bbox_inches="tight")

    # 5. IDF profiles across strategies
    fig5 = plot_strategy_idf_profiles(loinc, STRATEGIES_TO_COMPARE, rn_stopwords)
    fig5.savefig("data/results/viz_idf_profiles.png", dpi=150, bbox_inches="tight")

    # 6. UMAP of corpus geometry by strategy (colored by method_typ)
    fig6 = plot_tfidf_corpus_umap(
        loinc, STRATEGIES_TO_COMPARE, rn_stopwords, color_by="method_typ"
    )
    fig6.savefig("data/results/viz_corpus_umap.png", dpi=150, bbox_inches="tight")

    # 7. Joint UMAP: corpus docs + ELR queries
    fig7 = plot_tfidf_joint_umap(
        loinc,
        elr,
        corpus_strategy="lcn_method_dict_combined",
        rn_stopwords=rn_stopwords,
    )
    fig7.savefig("data/results/viz_joint_umap.png", dpi=150, bbox_inches="tight")

    # 8a. Full-corpus clustermap (98 codes, no labels — structure from sidebars)
    cg_full = plot_loinc_similarity_clustermap(
        loinc, corpus_strategy="lcn_method_dict_combined", rn_stopwords=rn_stopwords
    )
    cg_full.figure.savefig(
        "data/results/viz_similarity_clustermap_full.png", dpi=150, bbox_inches="tight"
    )

    # 8b. Eval-codes-only clustermap (13 codes, labeled + annotated — portfolio figure)
    # Provide the 13 loinc_nums that have seeds from your ablation run
    # eval_loincs = list(elr["loinc_num"].unique())
    # cg_eval = plot_loinc_similarity_clustermap(
    #     loinc, corpus_strategy="lcn_method_dict_combined",
    #     rn_stopwords=rn_stopwords, eval_loincs=eval_loincs
    # )
    # cg_eval.figure.savefig("data/results/viz_similarity_clustermap_eval.png", dpi=150, bbox_inches="tight")

    # 8b. Within/between — raw ratio (strategy comparison, TF-IDF only)
    fig8b = plot_within_between_similarity(
        loinc, STRATEGIES_TO_COMPARE, rn_stopwords, group_by="method_typ"
    )
    fig8b.savefig(
        "data/results/viz_within_between_sim.png", dpi=150, bbox_inches="tight"
    )

    # 8c. Within/between — Cohen's d (cross-method comparable)
    fig8c = plot_within_between_similarity(
        loinc,
        STRATEGIES_TO_COMPARE,
        rn_stopwords,
        group_by="method_typ",
        effect_size=True,
    )
    fig8c.savefig(
        "data/results/viz_within_between_cohens_d.png", dpi=150, bbox_inches="tight"
    )

    # 8d. ST within/between (requires sentence-transformers)
    # ST_MODELS = [
    #     "sentence-transformers/all-MiniLM-L6-v2",
    #     "sentence-transformers/all-mpnet-base-v2",
    # ]
    # fig8d = plot_st_within_between_similarity(
    #     loinc, ST_MODELS, group_by="method_typ", effect_size=True
    # )
    # fig8d.savefig("data/results/viz_st_within_between_cohens_d.png", dpi=150, bbox_inches="tight")

    # 8e. Similarity distribution shapes: TF-IDF vs ST — single clean call
    # (requires sentence-transformers installed)
    # fig8e = plot_similarity_distributions_from_names(
    #     loinc,
    #     tfidf_strategy="lcn_method_dict_combined",
    #     st_model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     rn_stopwords=rn_stopwords,
    #     group_by="method_typ",
    # )
    # fig8e.savefig("data/results/viz_sim_distributions.png", dpi=150, bbox_inches="tight")

    # 9. ELR length distributions
    fig9 = plot_elr_length_distribution(elr)
    fig9.savefig("data/results/viz_elr_lengths.png", dpi=150, bbox_inches="tight")

    # 10. Collision audit (text report)
    plot_elr_token_collision_audit(elr)

    # 11. Sentence transformer UMAP (requires sentence-transformers installed)
    # fig_st = plot_st_corpus_umap(loinc, elr, color_by="method_typ")
    # fig_st.savefig("data/results/viz_st_umap.png", dpi=150, bbox_inches="tight")
