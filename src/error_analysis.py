# src/error_analysis.py
# Error analysis and visualization for LOINC retrieval results.
# Works with any results DataFrame produced by evaluate_pipeline (TF-IDF or ST).
#
# Required columns in df_results:
#   true_loinc, predicted_loinc, mrr_grouped, top1,
#   coverage_pattern, corpus_strategy (or model_desc for labeling)
#   noise_corruption, noise_compression, noise_omission, noise_level


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional

from src.model_building_utils import CATCHALL_COVERAGE, GENE_TARGET_AMBIGUITY_LOOKUP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATCHALL_SYSTEMS = set(CATCHALL_COVERAGE.keys())

LOINC_AXES = ["component", "method_typ", "system"]

# Coverage patterns ordered by information content
COVERAGE_ORDER = [
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

NOISE_LEVEL_ORDER = ["low", "medium", "high"]

NOISE_COLS = ["noise_corruption", "noise_compression", "noise_omission"]

# Palette shared across comparison plots for consistency
MODEL_PALETTE = {
    "tfidf": "#2ecc71",
    "sentence_transformer": "#3498db",
}


# ---------------------------------------------------------------------------
# Step 1: Classify system error direction
# ---------------------------------------------------------------------------


def classify_system_direction(true_sys: str, pred_sys: str) -> str:
    """
    Classifies the directionality of a system axis difference.

      same:                 no system difference
      catchall_to_specific: true=catchall, pred=specific — corpus artifact,
                            absorbed by grouped MRR in most cases
      specific_to_catchall: true=specific, pred=catchall — retrieval failure
                            when mrr_grouped=0, specimen signal absent
      specific_to_specific: both specific but different (e.g. Nph vs Nose)
    """
    if true_sys == pred_sys:
        return "same"
    if true_sys in CATCHALL_SYSTEMS and pred_sys not in CATCHALL_SYSTEMS:
        return "catchall_to_specific"
    if true_sys not in CATCHALL_SYSTEMS and pred_sys in CATCHALL_SYSTEMS:
        return "specific_to_catchall"
    return "specific_to_specific"


# ---------------------------------------------------------------------------
# Step 2: Axis mismatch classifier
# ---------------------------------------------------------------------------


def classify_axis_mismatches(
    df_results: pd.DataFrame, df_loinc: pd.DataFrame
) -> pd.DataFrame:
    """
    For each wrong top-1 prediction (top1=0), classifies which LOINC axes
    the predicted code misaligns on relative to the true code.

    System mismatch uses equivalence-aware direction logic:
      - catchall_to_specific errors absorbed by mrr_grouped are NOT flagged
        as system mismatches — they are corpus artifacts, not retrieval failures
      - specific_to_catchall with mrr_grouped=0 ARE flagged
      - specific_to_specific always flagged

    Returns df_results with added columns:
      direction_system, component_mismatch, method_mismatch, system_mismatch
    """
    loinc_ref = df_loinc.set_index("loinc_num")[LOINC_AXES].to_dict(orient="index")

    def get_axis(code, axis):
        return loinc_ref.get(code, {}).get(axis, None)

    df = df_results.copy()
    wrong = df["top1"] == 0

    df["true_component"] = df["true_loinc"].map(lambda c: get_axis(c, "component"))
    df["pred_component"] = df["predicted_loinc"].map(lambda c: get_axis(c, "component"))
    df["true_method_typ"] = df["true_loinc"].map(lambda c: get_axis(c, "method_typ"))
    df["pred_method_typ"] = df["predicted_loinc"].map(
        lambda c: get_axis(c, "method_typ")
    )
    df["true_system"] = df["true_loinc"].map(lambda c: get_axis(c, "system"))
    df["pred_system"] = df["predicted_loinc"].map(lambda c: get_axis(c, "system"))

    df["direction_system"] = df.apply(
        lambda r: (
            classify_system_direction(r["true_system"], r["pred_system"])
            if wrong[r.name]
            else "correct"
        ),
        axis=1,
    )

    def component_mismatch(row):
        if not wrong[row.name]:
            return False
        if row["true_component"] == row["pred_component"]:
            return False
        true_group = GENE_TARGET_AMBIGUITY_LOOKUP.get(row["true_loinc"], set())
        if row["predicted_loinc"] in true_group:
            return False
        return True

    df["component_mismatch"] = df.apply(component_mismatch, axis=1)
    df["method_mismatch"] = wrong & (df["true_method_typ"] != df["pred_method_typ"])

    def system_mismatch(row):
        if not wrong[row.name]:
            return False
        direction = row["direction_system"]
        if direction == "same":
            return False
        if direction == "catchall_to_specific" and row["mrr_grouped"] >= 1.0:
            return False
        return True

    df["system_mismatch"] = df.apply(system_mismatch, axis=1)

    return df


# ---------------------------------------------------------------------------
# Step 3: Summary tables
# ---------------------------------------------------------------------------


def system_direction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Panel 1 data: system error direction x mrr_grouped correctness.
    Restricted to wrong top-1 predictions (top1==0).
    Retains both absorbed and real errors by design — this panel
    explicitly characterises the catchall/specific corpus artifact.
    """
    wrong = df[df["top1"] == 0].copy()
    wrong["mrr_correct"] = wrong["mrr_grouped"] >= 1.0
    return (
        wrong.groupby(["direction_system", "mrr_correct"])
        .size()
        .unstack("mrr_correct")
        .fillna(0)
        .astype(int)
        .rename(
            columns={
                True: "mrr_grouped=1 (absorbed)",
                False: "mrr_grouped<1 (real error)",
            }
        )
    )


def axis_mismatch_by_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Panel 2 data: % of all rows in each coverage pattern with each axis
    mismatched in the top-1 prediction.

    Denominator is all rows per coverage pattern (not filtered to failures).
    Since mismatch flags are False for correct predictions by construction,
    the mean naturally captures: what fraction of all rows in this pattern
    fail due to this axis? This avoids the compositional artifact where
    conditioning on failures inflates mismatch rates for axes that are
    incidental to the failure rather than causal.

    Panel 1 (system_direction_summary) retains top1==0 filtering because
    it explicitly characterises the catchall/specific corpus artifact and
    must operate on wrong predictions by definition.
    """
    result = (
        df.groupby("coverage_pattern")[
            ["component_mismatch", "method_mismatch", "system_mismatch"]
        ]
        .mean()
        .rename(
            columns={
                "component_mismatch": "Component",
                "method_mismatch": "Method",
                "system_mismatch": "System",
            }
        )
        * 100
    )
    present = [p for p in COVERAGE_ORDER if p in result.index]
    return result.reindex(present)


# def axis_mismatch_by_coverage(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Panel 2 data: % of genuine retrieval failures with each axis mismatched,
#     broken down by coverage_pattern.
#     Filters on mrr_grouped < 1 to exclude catchall/specific corpus artifacts
#     absorbed by equivalence grouping.
#     """
#     failures = df[df["mrr_grouped"] < 1].copy()
#     result = (
#         failures.groupby("coverage_pattern")[
#             ["component_mismatch", "method_mismatch", "system_mismatch"]
#         ]
#         .mean()
#         .rename(
#             columns={
#                 "component_mismatch": "Component",
#                 "method_mismatch": "Method",
#                 "system_mismatch": "System",
#             }
#         )
#         * 100
#     )
#     present = [p for p in COVERAGE_ORDER if p in result.index]
#     return result.reindex(present)


def axis_mismatch_by_strategy(
    df: pd.DataFrame, strategy_col: str = "corpus_strategy"
) -> pd.DataFrame:
    """
    Panel 3 data: axis mismatch rate by ablation strategy.

    Denominator is all rows per strategy (not filtered to failures).
    Since mismatch flags are False for correct predictions by construction,
    the mean captures: what fraction of all rows under this strategy fail
    due to each axis? Directly comparable across strategies without
    denominator instability from varying failure counts.
    """
    return (
        df.groupby(strategy_col)[
            ["component_mismatch", "method_mismatch", "system_mismatch"]
        ]
        .mean()
        .rename(
            columns={
                "component_mismatch": "Component",
                "method_mismatch": "Method",
                "system_mismatch": "System",
            }
        )
        * 100
    ).sort_values("Method", ascending=False)


# def axis_mismatch_by_strategy(
#     df: pd.DataFrame, strategy_col: str = "corpus_strategy"
# ) -> pd.DataFrame:
#     """
#     Panel 3 data: axis mismatch rate by ablation strategy.
#     Filters on mrr_grouped < 1 to exclude absorbed corpus artifacts.
#     """
#     failures = df[df["mrr_grouped"] < 1].copy()
#     return (
#         failures.groupby(strategy_col)[
#             ["component_mismatch", "method_mismatch", "system_mismatch"]
#         ]
#         .mean()
#         .rename(
#             columns={
#                 "component_mismatch": "Component",
#                 "method_mismatch": "Method",
#                 "system_mismatch": "System",
#             }
#         )
#         * 100
#     ).sort_values("Method", ascending=False)


def summarize_by_noise(df: pd.DataFrame, mrr_col: str = "mrr_grouped") -> pd.DataFrame:
    """
    MRR by noise_level stratum.
    Useful for confirming that relabelled noise thresholds produce
    stable per-stratum estimates.
    """
    return (
        df.groupby("noise_level")[mrr_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mrr_grouped_mean", "count": "n"})
        .reindex([l for l in NOISE_LEVEL_ORDER if l in df["noise_level"].unique()])
    )


# ---------------------------------------------------------------------------
# Step 4: Noise component analysis
# ---------------------------------------------------------------------------


def plot_noise_mrr_profile(
    df: pd.DataFrame,
    title_suffix: str = "",
    figsize: tuple = (18, 5),
) -> plt.Figure:
    """
    Three-panel figure showing mean grouped MRR as a function of each
    noise component count (corruption, compression, omission).

    Designed to show how each noise type independently degrades retrieval,
    holding the others unrestricted. Each panel plots mean mrr_grouped on
    the y-axis against the integer count of that noise type on the x-axis,
    with error bars (±1 SE) and n annotated above each bar.

    Parameters
    ----------
    df           : classified results DataFrame (output of classify_axis_mismatches)
    title_suffix : appended to figure suptitle (e.g. "— TF-IDF" or "— MiniLM")
    figsize      : figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        f"Grouped MRR by Noise Component Count {title_suffix}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    noise_labels = {
        "noise_corruption": "Corruption\n(character-level typos)",
        "noise_compression": "Compression\n(abbreviation / substitution)",
        "noise_omission": "Omission\n(token / component deletion)",
    }

    for ax, col in zip(axes, NOISE_COLS):
        grp = (
            df.groupby(col)["mrr_grouped"]
            .agg(["mean", "sem", "count"])
            .reset_index()
            .rename(columns={"mean": "mrr", "sem": "se", "count": "n"})
        )
        ax.bar(
            grp[col],
            grp["mrr"],
            yerr=grp["se"],
            color="#3498db",
            alpha=0.8,
            edgecolor="white",
            capsize=4,
        )
        for _, row in grp.iterrows():
            ax.text(
                row[col],
                row["mrr"] + row["se"] + 0.01,
                f"n={int(row['n'])}",
                ha="center",
                fontsize=8,
                color="#2c3e50",
            )
        ax.set_xlabel(f"{col} count", fontsize=10)
        ax.set_ylabel("Mean Grouped MRR", fontsize=10)
        ax.set_title(noise_labels[col], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(grp[col].astype(int))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Step 5: Cross-model comparison plots
# ---------------------------------------------------------------------------


def plot_model_comparison_by_coverage(
    df_tfidf: pd.DataFrame,
    df_st: pd.DataFrame,
    tfidf_label: str = "TF-IDF (best)",
    st_label: str = "Sentence Transformer (best)",
    mrr_col: str = "mrr_grouped",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Grouped bar chart comparing mean grouped MRR across coverage patterns
    for the best TF-IDF config and the best sentence transformer config.

    Each DataFrame should be pre-filtered to the single best configuration
    before passing (e.g. filter on corpus_strategy and model_desc).

    Parameters
    ----------
    df_tfidf   : results DataFrame for best TF-IDF config
    df_st      : results DataFrame for best sentence transformer config
    tfidf_label: legend label for TF-IDF bars
    st_label   : legend label for ST bars
    mrr_col    : MRR column to plot (default: mrr_grouped)
    figsize    : figure size
    """
    tfidf_cov = df_tfidf.groupby("coverage_pattern")[mrr_col].mean().rename(tfidf_label)
    st_cov = df_st.groupby("coverage_pattern")[mrr_col].mean().rename(st_label)

    combined = pd.concat([tfidf_cov, st_cov], axis=1).fillna(0)
    present = [p for p in COVERAGE_ORDER if p in combined.index]
    combined = combined.reindex(present)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(combined))
    width = 0.38

    bars_tfidf = ax.bar(
        x - width / 2,
        combined[tfidf_label],
        width=width,
        label=tfidf_label,
        color=MODEL_PALETTE["tfidf"],
        alpha=0.85,
        edgecolor="white",
    )
    bars_st = ax.bar(
        x + width / 2,
        combined[st_label],
        width=width,
        label=st_label,
        color=MODEL_PALETTE["sentence_transformer"],
        alpha=0.85,
        edgecolor="white",
    )

    for bars in [bars_tfidf, bars_st]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#2c3e50",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(combined.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean Grouped MRR", fontsize=11)
    ax.set_title(
        "Grouped MRR by Coverage Pattern\nTF-IDF vs Sentence Transformer",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="#95a5a6", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_model_comparison_by_noise(
    df_tfidf: pd.DataFrame,
    df_st: pd.DataFrame,
    tfidf_label: str = "TF-IDF (best)",
    st_label: str = "Sentence Transformer (best)",
    mrr_col: str = "mrr_grouped",
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    2x2 figure comparing the two best models across noise dimensions:

    Panel 1 (top-left):  Grouped MRR by noise_level (low/medium/high)
    Panel 2 (top-right): Grouped MRR by noise_corruption count
    Panel 3 (bottom-left):  Grouped MRR by noise_compression count
    Panel 4 (bottom-right): Grouped MRR by noise_omission count

    Each panel shows side-by-side bars for TF-IDF and ST.
    n is annotated above each bar group for transparency.

    Parameters
    ----------
    df_tfidf   : results DataFrame for best TF-IDF config
    df_st      : results DataFrame for best sentence transformer config
    tfidf_label: legend label for TF-IDF bars
    st_label   : legend label for ST bars
    mrr_col    : MRR column to plot
    figsize    : figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Grouped MRR by Noise Dimension\nTF-IDF vs Sentence Transformer",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    def _grouped_mean(df, groupby_col):
        return df.groupby(groupby_col)[mrr_col].agg(["mean", "count"]).reset_index()

    def _draw_panel(ax, col, order=None, xlabel=None):
        g_tfidf = _grouped_mean(df_tfidf, col)
        g_st = _grouped_mean(df_st, col)

        if order is not None:
            g_tfidf = g_tfidf.set_index(col).reindex(order).reset_index().dropna()
            g_st = g_st.set_index(col).reindex(order).reset_index().dropna()

        x = np.arange(len(g_tfidf))
        width = 0.38

        bars_t = ax.bar(
            x - width / 2,
            g_tfidf["mean"],
            width=width,
            label=tfidf_label,
            color=MODEL_PALETTE["tfidf"],
            alpha=0.85,
            edgecolor="white",
        )
        bars_s = ax.bar(
            x + width / 2,
            g_st["mean"],
            width=width,
            label=st_label,
            color=MODEL_PALETTE["sentence_transformer"],
            alpha=0.85,
            edgecolor="white",
        )

        # Annotate n above each bar pair using TF-IDF counts (same ELR rows)
        for i, (_, row) in enumerate(g_tfidf.iterrows()):
            ax.text(
                i,
                max(
                    g_tfidf["mean"].iloc[i],
                    g_st["mean"].iloc[i] if i < len(g_st) else 0,
                )
                + 0.03,
                f"n={int(row['count'])}",
                ha="center",
                fontsize=7,
                color="#2c3e50",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(g_tfidf[col].astype(str), fontsize=9)
        ax.set_xlabel(xlabel or col, fontsize=10)
        ax.set_ylabel("Mean Grouped MRR", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="#95a5a6", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

    present_levels = [
        l for l in NOISE_LEVEL_ORDER if l in df_tfidf["noise_level"].unique()
    ]
    _draw_panel(axes[0, 0], "noise_level", order=present_levels, xlabel="Noise Level")
    axes[0, 0].set_title("By Noise Level", fontsize=11)

    _draw_panel(axes[0, 1], "noise_corruption", xlabel="Corruption Count")
    axes[0, 1].set_title("By Corruption Count\n(character-level typos)", fontsize=11)

    _draw_panel(axes[1, 0], "noise_compression", xlabel="Compression Count")
    axes[1, 0].set_title(
        "By Compression Count\n(abbreviation / substitution)", fontsize=11
    )

    _draw_panel(axes[1, 1], "noise_omission", xlabel="Omission Count")
    axes[1, 1].set_title("By Omission Count\n(token / component deletion)", fontsize=11)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Step 6: Per-model error analysis figure
# ---------------------------------------------------------------------------


def plot_error_analysis(
    df: pd.DataFrame,
    df_loinc: pd.DataFrame,
    strategy_col: str = "corpus_strategy",
    title_suffix: str = "",
    figsize: tuple = (18, 14),
) -> plt.Figure:
    """
    Three-panel error analysis figure for a single model/config.

    Panel 1 (top-left): System error direction stacked bar.
      Filtered on top1==0 — retains both absorbed and real errors
      to demonstrate equivalence grouping validity.

    Panel 2 (top-right): Axis mismatch rate heatmap by coverage pattern.
      Filtered on mrr_grouped < 1 — genuine retrieval failures only.

    Panel 3 (bottom): Axis mismatch rate by ablation strategy, grouped bar.
      Filtered on mrr_grouped < 1 — genuine retrieval failures only.

    Parameters
    ----------
    df           : pre-classified results DataFrame (output of classify_axis_mismatches)
    df_loinc     : LOINC reference table (36-code panel)
    strategy_col : column for Panel 3 grouping
    title_suffix : appended to suptitle
    figsize      : figure size
    """
    dir_summary = system_direction_summary(df)
    coverage_tbl = axis_mismatch_by_coverage(df)
    strategy_tbl = axis_mismatch_by_strategy(df, strategy_col=strategy_col)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # ------------------------------------------------------------------
    # Panel 1: System error direction stacked bar (top1==0)
    # ------------------------------------------------------------------
    colors = {
        "mrr_grouped=1 (absorbed)": "#2ecc71",
        "mrr_grouped<1 (real error)": "#e74c3c",
    }
    dir_summary.plot(
        kind="barh",
        stacked=True,
        ax=ax1,
        color=[colors.get(c, "#95a5a6") for c in dir_summary.columns],
        edgecolor="white",
        linewidth=0.6,
    )
    ax1.set_title(
        "System Axis Error Direction\n(wrong top-1 predictions, top1=0)",
        fontsize=11,
    )
    ax1.set_xlabel("Count")
    ax1.set_ylabel("")
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.8)

    for i, (idx, row) in enumerate(dir_summary.iterrows()):
        total = row.sum()
        ax1.text(
            total + 1, i, str(int(total)), va="center", fontsize=8, color="#2c3e50"
        )

    # ------------------------------------------------------------------
    # Panel 2: Axis mismatch heatmap by coverage pattern (mrr_grouped < 1)
    # ------------------------------------------------------------------
    if not coverage_tbl.empty:
        sns.heatmap(
            coverage_tbl,
            ax=ax2,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
            linewidths=0.4,
            #            cbar_kws={"label": "% genuine failures\nwith axis mismatch"},
            cbar_kws={"label": "% all rows \nwith axis mismatch"},
        )
        ax2.set_title(
            "Axis Mismatch Rate by Coverage Pattern",
            fontsize=11,
        )

        # ax2.set_title(
        #     "Axis Mismatch Rate by Coverage Pattern\n(genuine failures: mrr_grouped < 1)",
        #     fontsize=11,
        # )
        ax2.set_xlabel("LOINC Axis")
        ax2.set_ylabel("Coverage Pattern")
        ax2.tick_params(axis="x", rotation=0)
        ax2.tick_params(axis="y", rotation=0)
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center")
        ax2.set_title("Axis Mismatch Rate by Coverage Pattern")

    # ------------------------------------------------------------------
    # Panel 3: Axis mismatch rate by strategy (mrr_grouped < 1)
    # ------------------------------------------------------------------
    x = np.arange(len(strategy_tbl))
    width = 0.25
    axis_colors = {"Component": "#3498db", "Method": "#e67e22", "System": "#9b59b6"}

    for i, (axis_label, color) in enumerate(axis_colors.items()):
        bars = ax3.bar(
            x + (i - 1) * width,
            strategy_tbl[axis_label],
            width=width,
            label=axis_label,
            color=color,
            alpha=0.85,
            edgecolor="white",
        )
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.5,
                    f"{h:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#2c3e50",
                )

    ax3.set_xticks(x)
    ax3.set_xticklabels(strategy_tbl.index, rotation=25, ha="right", fontsize=9)
    ax3.set_ylabel("% of genuine failures\nwith axis mismatch")
    ax3.set_title(
        "Axis Mismatch Rate by Retrieval Strategy\n(genuine failures: mrr_grouped < 1)",
        fontsize=11,
    )
    ax3.legend(title="LOINC Axis", fontsize=9)
    ax3.set_ylim(0, min(100, strategy_tbl.values.max() * 1.2))
    ax3.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax3.set_axisbelow(True)

    fig.suptitle(
        f"LOINC Retrieval Error Analysis {title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    return fig


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------


def run_error_analysis(
    df_results: pd.DataFrame,
    df_loinc: pd.DataFrame,
    strategy_col: str = "corpus_strategy",
    title_suffix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Full single-model error analysis pipeline: classify mismatches then plot.
    Call identically for TF-IDF and sentence transformer results.

    Parameters
    ----------
    df_results   : output of evaluate_pipeline
    df_loinc     : LOINC reference table (36-code panel)
    strategy_col : column for Panel 3 grouping
                   ("corpus_strategy" for TF-IDF, "model_desc" for ST)
    title_suffix : e.g. "— TF-IDF (best config)" or "— MiniLM"
    save_path    : if provided, saves figure to this path
    """
    df_classified = classify_axis_mismatches(df_results, df_loinc)
    fig = plot_error_analysis(
        df_classified,
        df_loinc,
        strategy_col=strategy_col,
        title_suffix=title_suffix,
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # prevent duplicate render in Jupyter
    return fig


def run_noise_analysis(
    df_results: pd.DataFrame,
    df_loinc: pd.DataFrame,
    title_suffix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Noise component analysis for a single model/config.
    Classifies mismatches then plots grouped MRR against each noise
    component count (corruption, compression, omission).

    Parameters
    ----------
    df_results   : output of evaluate_pipeline
    df_loinc     : LOINC reference table
    title_suffix : appended to figure suptitle
    save_path    : if provided, saves figure to this path
    """
    df_classified = classify_axis_mismatches(df_results, df_loinc)
    fig = plot_noise_mrr_profile(df_classified, title_suffix=title_suffix)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def run_model_comparison(
    df_tfidf: pd.DataFrame,
    df_st: pd.DataFrame,
    df_loinc: pd.DataFrame,
    tfidf_label: str = "TF-IDF (best)",
    st_label: str = "Sentence Transformer (best)",
    save_path_coverage: Optional[str] = None,
    save_path_noise: Optional[str] = None,
) -> tuple:
    """
    Cross-model comparison pipeline. Produces two figures:
      1. Grouped MRR by coverage pattern (side-by-side bars)
      2. Grouped MRR by noise dimension — noise_level, corruption,
         compression, omission (2x2 panel)

    Both DataFrames should be pre-filtered to the single best configuration
    for each model type before passing.

    Parameters
    ----------
    df_tfidf            : best TF-IDF config results
    df_st               : best ST config results
    df_loinc            : LOINC reference table
    tfidf_label         : legend label for TF-IDF
    st_label            : legend label for ST
    save_path_coverage  : optional save path for coverage figure
    save_path_noise     : optional save path for noise figure

    Returns
    -------
    (fig_coverage, fig_noise) — both figures, closed after save if path provided
    """
    df_tfidf_c = classify_axis_mismatches(df_tfidf, df_loinc)
    df_st_c = classify_axis_mismatches(df_st, df_loinc)

    fig_coverage = plot_model_comparison_by_coverage(
        df_tfidf_c,
        df_st_c,
        tfidf_label=tfidf_label,
        st_label=st_label,
    )
    fig_noise = plot_model_comparison_by_noise(
        df_tfidf_c,
        df_st_c,
        tfidf_label=tfidf_label,
        st_label=st_label,
    )

    if save_path_coverage:
        fig_coverage.savefig(save_path_coverage, dpi=150, bbox_inches="tight")
    if save_path_noise:
        fig_noise.savefig(save_path_noise, dpi=150, bbox_inches="tight")

    plt.close(fig_coverage)
    plt.close(fig_noise)
    return fig_coverage, fig_noise
