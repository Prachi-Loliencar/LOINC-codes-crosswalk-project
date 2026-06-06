# src/model_building_utils.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit
import re
from dataclasses import dataclass
from typing import Literal, Optional
from sklearn.preprocessing import normalize
from scipy.sparse import hstack


from src.clinical_utils import (
    LOINC_SYSTEM_EXPANSION,
    LOINC_METHOD_EXPANSION,
    clean_text,
    SPECIMEN_TO_LOINC_SYSTEM,
)

# ---------------------------------------------------------------------------
# LOINC LCN expansion — retrieval-side only, never used in simulation
# ---------------------------------------------------------------------------

LOINC_LCN_EXPANSION = [
    (r"\bPRESENCE\b", ""),
    (r"\bIDENTIFIED\s+IN\b", "IN"),
    (r"\bPANEL\b", ""),
    (r"\bDNA\s+AND\s+RNA\b", "RNA DNA"),
    (r"\bUNITS\s*VOLUME\b", "QUANT"),
    (r"\bVIRAL\s+LOAD\b", "QUANT"),
    (
        r"\bNAA\s+WITH\s+NON\s*PROBE\s+DETECTION\b",
        "NAA WITH NON PROBE DETECTION NAAT NAA",
    ),
    (r"\bNAA\s+WITH\s+PROBE\s+DETECTION\b", "NAA WITH PROBE DETECTION NAAT NAA PCR"),
    (
        r"\bNUCLEIC\s+ACID\s+AMPLIFICATION\s+USING\s+CDC\s+PRIMER\s+PROBE\s+SET\s+N1\b",
        "NUCLEIC ACID AMPLIFICATION CDC PRIMER PROBE SET N1 NAAT NAA PCR",
    ),
    (
        r"\bNUCLEIC\s+ACID\s+AMPLIFICATION\s+USING\s+CDC\s+PRIMER\s+PROBE\s+SET\s+N2\b",
        "NUCLEIC ACID AMPLIFICATION CDC PRIMER PROBE SET N2 NAAT NAA PCR",
    ),
    (r"\bRAPID\s+IMMUNOASSAY\b", "RAPID IMMUNOASSAY RAPID ANTIGEN AG"),
    (r"\bIMMUNOASSAY\b", "IMMUNOASSAY ANTIGEN AG ELISA"),
    (
        r"\bSARS\s+RELATED\s+CORONAVIRUS\b",
        "SARS RELATED CORONAVIRUS SARSREL SARBECOVIRUS SARSCOV2",
    ),
    (r"\bSARS\s+COV\s+2\b", "SARS COV 2 SARSCOV2"),
    (r"\bN\s+GENE\b", "N GENE NGENE"),
    (r"\bRDRP\s+GENE\b", "RDRP GENE RDRP"),
    (r"\bORF1AB\s+REGION\b", "ORF1AB REGION ORF1AB"),
    (r"\bORF1A\s+REGION\b", "ORF1A REGION ORF1A"),
    (r"\bS\s+GENE\b", "S GENE SGENE"),
    (r"\bE\s+GENE\b", "E GENE EGENE"),
    (r"\bM\s+GENE\b", "M GENE MGENE"),
    (r"\bNSP2\s+GENE\b", "NSP2 GENE NSP2"),
    (
        r"\bRESPIRATORY\s+SYNCYTIAL\s+VIRUS\s+A\b",
        "RESPIRATORY SYNCYTIAL VIRUS A RSV RSVA",
    ),
    (
        r"\bRESPIRATORY\s+SYNCYTIAL\s+VIRUS\s+B\b",
        "RESPIRATORY SYNCYTIAL VIRUS B RSV RSVB",
    ),
    (r"\bRESPIRATORY\s+SYNCYTIAL\s+VIRUS\b", "RESPIRATORY SYNCYTIAL VIRUS RSV"),
    (r"\bHUMAN\s+METAPNEUMOVIRUS\b", "HUMAN METAPNEUMOVIRUS HMPV MPV"),
    (r"\bPARAINFLUENZA\s+VIRUS\s+1\b", "PARAINFLUENZA VIRUS 1 PIV1 PIV"),
    (r"\bPARAINFLUENZA\s+VIRUS\s+2\b", "PARAINFLUENZA VIRUS 2 PIV2 PIV"),
    (r"\bPARAINFLUENZA\s+VIRUS\s+3\b", "PARAINFLUENZA VIRUS 3 PIV3 PIV"),
    (r"\bPARAINFLUENZA\s+VIRUS\s+4\b", "PARAINFLUENZA VIRUS 4 PIV4 PIV"),
    (r"\bPARAINFLUENZA\s+VIRUS\b", "PARAINFLUENZA VIRUS PIV"),
    (r"\bBORDETELLA\s+PERTUSSIS\b", "BORDETELLA PERTUSSIS BPERT"),
    (r"\bBORDETELLA\s+PARAPERTUSSIS\b", "BORDETELLA PARAPERTUSSIS BPARAPERT"),
    (r"\bADENOVIRUS\b", "ADENOVIRUS ADV ADENO"),
    (r"\bRHINOVIRUS\b", "RHINOVIRUS RV RHINO"),
    (r"\bENTEROVIRUS\b", "ENTEROVIRUS EV"),
    (r"\bHUMAN\s+CORONAVIRUS\s+HKU1\b", "HUMAN CORONAVIRUS HKU1 HCOV HCOVHKU1"),
    (r"\bHUMAN\s+CORONAVIRUS\s+NL63\b", "HUMAN CORONAVIRUS NL63 HCOV HCOVNL63"),
    (r"\bHUMAN\s+CORONAVIRUS\s+229E\b", "HUMAN CORONAVIRUS 229E HCOV HCOV229E"),
    (r"\bHUMAN\s+CORONAVIRUS\s+OC43\b", "HUMAN CORONAVIRUS OC43 HCOV HCOVOCC43"),
    (r"\bHUMAN\s+CORONAVIRUS\b", "HUMAN CORONAVIRUS HCOV"),
    (r"\bHUMAN\s+BOCAVIRUS\b", "HUMAN BOCAVIRUS HBOV BOCA"),
    (r"\bMYCOPLASMA\s+PNEUMONIAE\b", "MYCOPLASMA PNEUMONIAE MYPNEU"),
    (r"\bCHLAMYDOPHILA\s+PNEUMONIAE\b", "CHLAMYDOPHILA PNEUMONIAE CPNEU"),
    (r"\bINFLUENZA\s+VIRUS\b(?!\s+[AB]\b)", "INFLUENZA VIRUS FLUA FLUB"),
]

_LCN_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in LOINC_LCN_EXPANSION]


def expand_loinc_lcn(text: str) -> str:
    """
    Applies LCN expansion rules and deduplicates tokens preserving order.
    Retrieval-side only — must not be called in simulation code.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.upper()
    for pattern, replacement in _LCN_PATTERNS:
        text = pattern.sub(replacement, text)
    seen, tokens = set(), []
    for tok in text.split():
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# ELR query normalization — retrieval-side only
# ---------------------------------------------------------------------------

ELR_NORMALIZATION = [
    (r"\bSC2\b", "SARSCOV2"),
    (r"\bPAN[\s-]SARBECOVIRUS\b", "SARSREL SARBECOVIRUS SARSCOV2"),
    (r"\bSARBECOVIRUS\b", "SARSREL SARBECOVIRUS SARSCOV2"),
    (r"\bINFA\b", "FLUA"),
    (r"\bINFB\b", "FLUB"),
    (r"\bLOWER\s+RESPIRATORY\s+TRACT\b", "LRT BAL LOWER RESPIRATORY"),
    (r"\bINFLUENZA\b", "FLUA FLUB INFLUENZA"),
    (r"\bFLU\b", "FLUA FLUB FLU"),
    (r"\bRDRP\s+GENE\b", "RDRP GENE RDRP"),
]

_ELR_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in ELR_NORMALIZATION]

# Method tokens derived from LOINC method phrases — retrieval-side only.
LOINC_METHOD_TOKENS = {
    "Non-probe.amp.tar": "NAAT NAA NON PROBE",
    "Probe.amp.tar": "NAAT NAA PCR PROBE",
    "Probe.amp.tar.CDC primer-probe set N1": "NAAT NAA PCR PROBE N1",
    "Probe.amp.tar.CDC primer-probe set N2": "NAAT NAA PCR PROBE N2",
    "IA.rapid": "RAPID ANTIGEN AG IMMUNOASSAY",
    "IA": "ANTIGEN AG IMMUNOASSAY",
}

# High-confidence brand-to-method_class mappings.
# Two-token keys (e.g. "BD VERITOR") take priority over single-token keys.
# Ambiguous single-brand entries are omitted to avoid false imputation.
BRAND_METHOD_MAP = {
    # Antigen rapid
    "BINAXNOW": "ia.rapid",
    "SOFIA": "ia.rapid",
    "QUICKVUE": "ia.rapid",
    "FLOWFLEX": "ia.rapid",
    "PANBIO": "ia.rapid",
    "CARESTART": "ia.rapid",
    "INTELISWAB": "ia.rapid",
    "INDICAID": "ia.rapid",
    "CLINITEST": "ia.rapid",
    "IHEALTH": "ia.rapid",
    "OSOM": "ia.rapid",
    "HEALGEN": "ia.rapid",
    "HOTGEN": "ia.rapid",
    "VERITOR": "ia.rapid",  # standalone (not BD VERITOR)
    "BD VERITOR": "ia.rapid",  # two-token — checked first
    # IA (non-rapid)
    # "LUMIRADX": "ia",
    "LIAISON": "ia",
    "VITROS": "ia",
    # Probe amplification
    "COBAS": "probe.amp.tar",
    "TAQPATH": "probe.amp.tar",
    "ALINITY": "probe.amp.tar",
    "M2000": "probe.amp.tar",
    "APTIMA": "probe.amp.tar",
    "PANTHER": "probe.amp.tar",
    "NEUMODX": "probe.amp.tar",
    "ARIES": "probe.amp.tar",
    "QIASTAT": "probe.amp.tar",
    "ALLPLEX": "probe.amp.tar",
    "LYRA": "probe.amp.tar",
    "REALSTAR": "probe.amp.tar",
    "GENEFINDER": "probe.amp.tar",
    "POWERCHEK": "probe.amp.tar",
    "BIOFIRE": "probe.amp.tar",
    "NXTAG": "probe.amp.tar",
    "BD MAX": "probe.amp.tar",  # two-token — checked first
    # Non-probe amplification
    "XPERT": "non-probe.amp.tar",
    "LUCIRA": "non-probe.amp.tar",
    "VISBY": "non-probe.amp.tar",
    "IAMP": "non-probe.amp.tar",
    "SHERLOCK": "non-probe.amp.tar",
}

METHOD_TYP_TO_CLASS = {
    "probe.amp.tar": "naat",
    "probe.amp.tar.cdc primer-probe set n1": "naat",
    "probe.amp.tar.cdc primer-probe set n2": "naat",
    "non-probe.amp.tar": "naat",
    "ia.rapid": "antigen",
    "ia": "antigen",
    "if": "antigen",
}


def normalize_elr(text: str) -> str:
    """
    Apply ELR normalization at retrieval time.
    Assumes clean_text has already been applied.
    """
    if not text or not isinstance(text, str):
        return ""
    for pattern, replacement in _ELR_PATTERNS:
        text = pattern.sub(replacement, text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    filter_strategy: Literal["none", "method_only", "specimen_only", "both"]
    corpus_strategy: Literal[
        "lcn_only",
        "combined",
        "lcn_filtered_rn_combined",
        "lcn_method_dict_combined",
        "lcn_method_dict_filtered_rn",
        "component_weighted_method_dict",
    ]
    model_type: Literal["tfidf_word", "tfidf_char", "tfidf_mixed"]
    n_neighbors: int = 6
    extended_corpus: bool = False
    n_distractors: int = 0
    ngram_word: tuple = (1, 1)
    ngram_char: tuple = (3, 5)
    alpha: float = 0.5
    # Post-retrieval filter flags — False by default, enabled only in secondary ablation
    oracle_filter: bool = False  # Filter A: ground-truth method+specimen upper bound
    brand_filter: bool = False  # Filter B: imputes method from model name tokens


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def compute_relatednames_stopwords(
    df_loinc: pd.DataFrame, threshold: float = 0.85
) -> set:
    """Tokens in >threshold fraction of relatednames2 fields — suppressed as uninformative."""
    from collections import Counter

    RN_WHITELIST = {"NAAT", "NAA", "PCR"}
    clean_synonyms = (
        df_loinc["relatednames2"].fillna("").str.replace(";", " ").map(clean_text)
    )
    n_docs = len(df_loinc)
    doc_freq = Counter()
    for syn_str in clean_synonyms:
        doc_freq.update(set(syn_str.split()))
    return {
        tok
        for tok, freq in doc_freq.items()
        if freq / n_docs >= threshold and tok not in RN_WHITELIST
    }


def filter_relatednames(text: str, stopwords: set) -> str:
    if not text:
        return ""
    tokens = clean_text(text.replace(";", " ")).split()
    return " ".join(t for t in tokens if t not in stopwords)


def sample_distractors(sample_size: int, df_distractors: pd.DataFrame) -> pd.DataFrame:
    return (
        df_distractors.sample(min(sample_size, len(df_distractors)), random_state=42)
        if sample_size < len(df_distractors)
        else df_distractors
    )


def build_corpus(df_loinc: pd.DataFrame, strategy: str, rn_stopwords: set) -> pd.Series:
    """
    Returns a Series of preprocessed corpus strings aligned with df_loinc index.
    Requires expanded_lcn and relatednames2 columns.

    Strategies correspond exactly to those evaluated in the primary and
    secondary ablations. Two development-only strategies have been removed:

      axes_only  — system + relatednames2 with no LCN. Performance too low
                   to include; LCN is the primary discriminative signal.
      combined3  — 3x LCN repetition. Underperformed 2x repetition, confirming
                   that string repetition is an indirect substitute for explicit
                   vocabulary expansion rather than a reliable weighting mechanism.

    LCN repetition rationale (combined, lcn_method_dict_filtered_rn):
    The relatednames2 field is long and heterogeneous. Its tokens inflate raw
    TF counts and reduce the relative weight of discriminative LCN tokens.
    Repeating LCN raises its TF counts without affecting IDF, providing a
    partial counterbalance. lcn_method_dict_combined achieves higher performance
    without repetition by excluding relatednames2 entirely and replacing it with
    a compact method expansion dictionary, eliminating the dilution problem at
    its source rather than compensating for it indirectly.
    """
    lcn = df_loinc["expanded_lcn"]
    sys_exp = df_loinc["system"].map(
        lambda s: LOINC_SYSTEM_EXPANSION.get(str(s), str(s))
    )
    rn_raw = df_loinc["relatednames2"].fillna("").str.replace(";", " ")

    if strategy == "lcn_only":
        return lcn

    elif strategy == "combined":
        # LCN repeated 2x to counteract TF dilution from relatednames2.
        return lcn + " " + lcn + " " + sys_exp + " " + rn_raw

    elif strategy == "lcn_filtered_rn_combined":
        filtered_rn = rn_raw.map(lambda x: filter_relatednames(x, rn_stopwords))
        # LCN repeated 2x for same TF-dilution reason as combined.
        return lcn + " " + lcn + " " + sys_exp + " " + filtered_rn

    elif strategy == "lcn_method_dict_filtered_rn":
        meth_exp = df_loinc["method_typ"].map(
            lambda s: LOINC_METHOD_TOKENS.get(str(s), str(s))
        )
        filtered_rn = rn_raw.map(lambda x: filter_relatednames(x, rn_stopwords))
        # LCN repeated 2x for same TF-dilution reason as combined.
        return lcn + " " + lcn + " " + sys_exp + " " + meth_exp + " " + filtered_rn

    elif strategy == "lcn_method_dict_combined":
        meth_exp = df_loinc["method_typ"].map(
            lambda s: LOINC_METHOD_TOKENS.get(str(s), str(s))
        )
        # No LCN repetition needed — relatednames2 excluded entirely,
        # eliminating dilution at its source.
        return lcn + " " + sys_exp + " " + meth_exp

    elif strategy == "component_weighted_method_dict":
        comp_weighted = (
            df_loinc["component"].map(clean_text).map(expand_loinc_lcn) + " "
        ) * 2
        meth_exp = df_loinc["method_typ"].map(
            lambda m: LOINC_METHOD_EXPANSION.get(str(m), str(m))
        )
        return comp_weighted + " " + meth_exp + " " + sys_exp

    else:
        raise ValueError(f"Unknown corpus_strategy: {strategy}")


def check_distractors_idf(
    df_loinc: pd.DataFrame,
    df_distractors_noncovid: pd.DataFrame,
    rn_stopwords: set,
    corpus_strategy: str = "lcn_only",
) -> None:
    """Prints SARSCOV2/flu/method IDF at several distractor counts to verify monotonicity."""
    for n in [36, 50, 100, len(df_distractors_noncovid)]:
        df_sample = sample_distractors(n, df_distractors_noncovid)
        df_corpus = pd.concat([df_loinc, df_sample], ignore_index=True)
        corpus = build_corpus(df_corpus, corpus_strategy, rn_stopwords)
        vec = TfidfVectorizer(analyzer="word")
        vec.fit(corpus)
        vocab, idf = vec.vocabulary_, vec.idf_
        tokens_to_check = ["sarscov2", "flua", "flub", "rsv", "naat", "pcr"]
        idf_vals = {
            tok: np.round(idf[vocab[tok]], 3) for tok in tokens_to_check if tok in vocab
        }
        print(f"n_distractors={n}: {idf_vals}")


# ---------------------------------------------------------------------------
# Equivalence group handling
# ---------------------------------------------------------------------------

CATCHALL_COVERAGE = {
    "Respiratory System Specimen": {
        "Nph",
        "Nose",
        "Throat",
        "Saliva",
        "BAL",
        "Sputum",
        "Respiratory system specimen.upper",
        "Respiratory System Specimen",
    },
    "Respiratory system specimen.upper": {
        "Nph",
        "Nose",
        "Throat",
        "Respiratory system specimen.upper",
    },
}

GENE_TARGET_AMBIGUOUS_GROUPS = [
    {
        "94533-7",
        "94559-2",
        "94756-4",
        "94757-2",
        "94500-6",
        "94759-8",
        "94845-5",
        "95406-5",
        "94760-6",
        "95409-9",
        "96448-6",
        "97104-4",
        "95425-5",
    },
    {"96123-5", "96091-4"},
    {"96122-7"},
]

GENE_TARGET_AMBIGUITY_LOOKUP: dict = {}
for _group in GENE_TARGET_AMBIGUOUS_GROUPS:
    for _code in _group:
        GENE_TARGET_AMBIGUITY_LOOKUP[_code] = _group


def get_valid_loincs(
    elr_row: pd.Series, df_loinc: pd.DataFrame, true_loinc: str
) -> set:
    """
    Returns the set of LOINC codes constituting a correct retrieval for this
    ELR row: specimen-aware axis expansion plus gene-target ambiguity grouping.
    """
    true_row = df_loinc[df_loinc["loinc_num"] == true_loinc]
    if true_row.empty:
        return {true_loinc}

    CATCHALL_SYSTEMS = CATCHALL_COVERAGE.keys()
    specimen_norm = elr_row.get("specimen_norm", "UNKNOWN")
    valid_systems = SPECIMEN_TO_LOINC_SYSTEM.get(specimen_norm)

    same_axis = df_loinc[
        (df_loinc["component"] == true_row.iloc[0]["component"])
        & (df_loinc["method_typ"] == true_row.iloc[0]["method_typ"])
    ]
    if valid_systems is None:
        specimen_valid = same_axis[same_axis["system"].isin(CATCHALL_SYSTEMS)]
    else:
        specimen_valid = same_axis[same_axis["system"].isin(valid_systems)]

    valid_codes = set(specimen_valid["loinc_num"].values) | {true_loinc}

    ambiguity_group = GENE_TARGET_AMBIGUITY_LOOKUP.get(true_loinc)
    if ambiguity_group:
        ambig = df_loinc[df_loinc["loinc_num"].isin(ambiguity_group)]
        if valid_systems is None:
            ambig = ambig[ambig["system"].isin(CATCHALL_SYSTEMS)]
        else:
            ambig = ambig[ambig["system"].isin(valid_systems)]
        valid_codes |= set(ambig["loinc_num"].values)

    return valid_codes


# ---------------------------------------------------------------------------
# Vectorizer and index
# ---------------------------------------------------------------------------


def build_vectorizer(model_type: str, ngram_range: tuple) -> TfidfVectorizer:
    if model_type == "tfidf_word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            sublinear_tf=True,
            max_features=500,
            min_df=1,
            max_df=0.85,
        )
    elif model_type == "tfidf_char":
        return TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True
        )
    else:
        raise ValueError(f"Vectorizer not applicable for model_type: {model_type}")


def build_tfidf_index(corpus: pd.Series, model_type: str, ngram_range: tuple):
    vec = build_vectorizer(model_type, ngram_range)
    return vec, vec.fit_transform(corpus)


def build_mixed_tfidf_index(
    corpus: pd.Series,
    ngram_word: tuple = (1, 1),
    ngram_char: tuple = (3, 5),
    alpha: float = 0.5,
):
    """
    Mixed word+char index. Sub-matrices are L2-normalized before scaling so
    alpha maps linearly to cosine-space contribution.
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_word,
        sublinear_tf=True,
        min_df=1,
        max_features=500,
        max_df=0.85,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngram_char, sublinear_tf=True
    )
    word_matrix = normalize(word_vec.fit_transform(corpus))
    char_matrix = normalize(char_vec.fit_transform(corpus))
    combined = hstack([word_matrix * np.sqrt(alpha), char_matrix * np.sqrt(1 - alpha)])
    return word_vec, char_vec, combined, alpha


def build_nn_index(matrix, n_neighbors: int) -> NearestNeighbors:
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(matrix)
    return nn


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    elr_str: str, vectorizer, df_loinc_subset: pd.DataFrame, nn
) -> pd.DataFrame:
    """
    Returns ranked candidates for elr_str.
    Expects clean_text and normalize_elr already applied to elr_str.
    """
    distances, indices = nn.kneighbors(vectorizer.transform([elr_str]))
    candidates = df_loinc_subset.iloc[indices[0]].copy()
    candidates["base_score"] = 1 - distances[0]
    candidates["rank"] = range(1, len(candidates) + 1)
    return candidates[
        [
            "loinc_num",
            "long_common_name",
            "expanded_lcn",
            "corpus_text",
            "base_score",
            "rank",
        ]
    ]


def retrieve_mixed(
    elr_str: str, word_vec, char_vec, df_loinc_subset: pd.DataFrame, nn, alpha: float
) -> pd.DataFrame:
    """Retrieval for the mixed word+char model. alpha must match build_mixed_tfidf_index."""
    word_q = normalize(word_vec.transform([elr_str]))
    char_q = normalize(char_vec.transform([elr_str]))
    query = hstack([word_q * alpha, char_q * (1 - alpha)])
    distances, indices = nn.kneighbors(query)
    candidates = df_loinc_subset.iloc[indices[0]].copy()
    candidates["base_score"] = 1 - distances[0]
    candidates["rank"] = range(1, len(candidates) + 1)
    return candidates[
        [
            "loinc_num",
            "long_common_name",
            "expanded_lcn",
            "corpus_text",
            "base_score",
            "rank",
        ]
    ]


def score_metrics(candidates: pd.DataFrame) -> dict:
    top_score = candidates.iloc[0]["base_score"]
    second_score = candidates.iloc[1]["base_score"] if len(candidates) > 1 else 0.0
    score_gap = top_score - second_score
    return {
        "predicted_loinc": candidates.iloc[0]["loinc_num"],
        "expanded_lcn": candidates.iloc[0]["expanded_lcn"],
        "top_score": round(top_score, 4),
        "score_gap": round(score_gap, 4),
    }


# ---------------------------------------------------------------------------
# Post-retrieval metadata filters
# ---------------------------------------------------------------------------


def apply_oracle_filter(
    candidates: pd.DataFrame,
    elr_row: pd.Series,
    demotion_factor: float = 0.5,
) -> pd.DataFrame:
    """
    Filter A — Oracle upper bound.

    Uses ground-truth method_class and specimen_norm from the simulated ELR
    row to demote mismatching candidates. Represents the ceiling gain from
    perfect metadata extraction — not achievable in production.

    Requires method_class and system columns on candidates (join these in
    evaluate_pipeline before calling).

    Population where this has impact: has_method=0 rows where TF-IDF cannot
    distinguish method from token signal alone.
    """
    candidates = candidates.copy()
    match = pd.Series(True, index=candidates.index)

    elr_method = elr_row.get("method_class", "")
    if (
        elr_method
        and elr_method not in ("", "unknown")
        and "method_class" in candidates.columns
    ):
        match &= candidates["method_class"] == elr_method

    elr_spec = elr_row.get("specimen_norm", "UNKNOWN")
    valid_systems = SPECIMEN_TO_LOINC_SYSTEM.get(elr_spec)
    if valid_systems is not None and "system" in candidates.columns:
        CATCHALL_SYSTEMS = set(CATCHALL_COVERAGE.keys())
        match &= candidates["system"].isin(valid_systems) | candidates["system"].isin(
            CATCHALL_SYSTEMS
        )

    candidates["oracle_match"] = match
    candidates["filter_score"] = np.where(
        match, candidates["base_score"], candidates["base_score"] * demotion_factor
    )
    return candidates.sort_values("filter_score", ascending=False).reset_index(
        drop=True
    )


def impute_method_from_brand(elr_str: str) -> Optional[str]:
    """
    Scans the ELR string directly for known brand tokens and returns the
    imputed method_class if a high-confidence mapping exists.
    Checks two-token keys before single-token keys.
    Returns None if no mapping found — no filter applied in that case.
    """
    if not elr_str or not isinstance(elr_str, str):
        return None
    tokens = elr_str.upper().split()
    for i in range(len(tokens) - 1):
        two_tok = f"{tokens[i]} {tokens[i + 1]}"
        if two_tok in BRAND_METHOD_MAP:
            return BRAND_METHOD_MAP[two_tok]
    for tok in tokens:
        if tok in BRAND_METHOD_MAP:
            return BRAND_METHOD_MAP[tok]
    return None


def apply_brand_filter(
    candidates: pd.DataFrame,
    elr_row: pd.Series,
    demotion_factor: float = 0.5,
) -> pd.DataFrame:
    """
    Filter B — Brand-based method imputation from ELR string.

    Scans elr_name_normalized directly for brand tokens. No simulation
    metadata (Model column, has_model flag) is used — only what a real
    inference pipeline would see.

    Population where this fires: ELR strings containing a brand token
    that maps unambiguously to a method_class. When no brand is found
    the function is a no-op (brand_match=None, scores unchanged).
    """
    candidates = candidates.copy()
    elr_str = str(elr_row.get("elr_name_normalized", elr_row.get("elr_name", "")))
    imputed_method = impute_method_from_brand(elr_str)

    if imputed_method is None:
        candidates["brand_match"] = None
        candidates["filter_score"] = candidates["base_score"]
        return candidates

    # Translate method_typ vocabulary to method_class vocabulary
    imputed_class = METHOD_TYP_TO_CLASS.get(imputed_method, imputed_method)

    match = (
        candidates["method_class"] == imputed_class
        if "method_class" in candidates.columns
        else pd.Series(False, index=candidates.index)
    )

    candidates["brand_match"] = match
    candidates["imputed_method"] = imputed_method
    candidates["filter_score"] = np.where(
        match, candidates["base_score"], candidates["base_score"] * demotion_factor
    )
    return candidates.sort_values("filter_score", ascending=False).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_mrr(candidates: pd.DataFrame, true_loinc: str) -> float:
    for i, row in enumerate(candidates.itertuples(), start=1):
        if row.loinc_num == true_loinc:
            return 1.0 / i
    return 0.0


def compute_mrr_grouped(
    candidates: pd.DataFrame,
    elr_row: pd.Series,
    true_loinc: str,
    df_loinc: pd.DataFrame,
) -> float:
    valid_codes = get_valid_loincs(elr_row, df_loinc, true_loinc)
    for i, row in enumerate(candidates.itertuples(), start=1):
        if row.loinc_num in valid_codes:
            return 1.0 / i
    return 0.0


# def random_baseline_mrr(n_corpus: int, group_sizes: pd.Series) -> float:
#     """Expected grouped MRR under uniform random ranking."""
#     return np.mean([1 / ((n_corpus + 1) / (g + 1)) for g in group_sizes.values])


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def evaluate_pipeline(
    config: PipelineConfig,
    df_loinc: pd.DataFrame,
    df_test: pd.DataFrame,
    ngram_range: Optional[tuple],
    df_loinc_distractors: Optional[pd.DataFrame] = None,
    rn_stopwords: set = set(),
) -> pd.DataFrame:
    """
    Runs retrieval for all rows in df_test under the given PipelineConfig.

    Filter dispatch (secondary ablation only):
      config.oracle_filter=True  → apply_oracle_filter post-retrieval
      config.brand_filter=True   → apply_brand_filter post-retrieval
      Both False (default)       → no post-retrieval filtering
    """
    # 1. Build corpus (extend with distractors if configured)
    if config.extended_corpus:
        if df_loinc_distractors is None:
            raise ValueError("extended_corpus=True requires df_loinc_distractors")
        df_dist = sample_distractors(config.n_distractors, df_loinc_distractors.copy())
        df_dist["is_eval_target"] = False
        df_loinc_corpus = pd.concat(
            [df_loinc.assign(is_eval_target=True), df_dist], ignore_index=True
        )
    else:
        df_loinc_corpus = df_loinc.assign(is_eval_target=True)

    corpus = build_corpus(df_loinc_corpus, config.corpus_strategy, rn_stopwords)
    df_loinc_corpus = df_loinc_corpus.copy()
    df_loinc_corpus["corpus_text"] = corpus

    # 2. Fit vectorizer
    alpha = None
    if config.model_type == "tfidf_mixed":
        word_vec, char_vec, corpus_matrix, alpha = build_mixed_tfidf_index(
            corpus,
            ngram_word=config.ngram_word,
            ngram_char=config.ngram_char,
            alpha=config.alpha,
        )
    else:
        vectorizer, corpus_matrix = build_tfidf_index(
            corpus, config.model_type, ngram_range
        )

    nn = build_nn_index(corpus_matrix, n_neighbors=config.n_neighbors)

    results = []
    for _, elr_row in df_test.iterrows():
        elr_str = elr_row["elr_name_normalized"]

        if config.model_type == "tfidf_mixed":
            candidates = retrieve_mixed(
                elr_str, word_vec, char_vec, df_loinc_corpus, nn, alpha=alpha
            )
        else:
            candidates = retrieve(elr_str, vectorizer, df_loinc_corpus, nn)
        scores = score_metrics(candidates)

        # Post-retrieval filtering (secondary ablation only)
        filter_applied = "none"
        if config.oracle_filter:
            candidates = candidates.merge(
                df_loinc_corpus[["loinc_num", "method_class", "system"]],
                on="loinc_num",
                how="left",
            )
            candidates = apply_oracle_filter(candidates, elr_row)
            filter_applied = "oracle"
        elif config.brand_filter:
            candidates = candidates.merge(
                df_loinc_corpus[["loinc_num", "method_class", "system"]],
                on="loinc_num",
                how="left",
            )
            candidates = apply_brand_filter(candidates, elr_row)
            filter_applied = "brand"

        mrr = compute_mrr(candidates, elr_row["loinc_num"])
        mrr_grouped = compute_mrr_grouped(
            candidates, elr_row, elr_row["loinc_num"], df_loinc
        )

        results.append(
            {
                **config.__dict__,
                "filter_applied": filter_applied,
                "elr_name": elr_row.get("elr_name"),
                "elr_name_normalized": elr_row.get("elr_name_normalized"),
                "true_loinc": elr_row.get("loinc_num"),
                "specimen_norm": elr_row.get("specimen_norm"),
                "noise_level": elr_row.get("noise_level"),
                "noise_compression": elr_row.get("noise_compression"),
                "noise_omission": elr_row.get("noise_omission"),
                "noise_corruption": elr_row.get("noise_corruption"),
                "noise_total": elr_row.get("noise_total"),
                "coverage_pattern": elr_row.get("coverage_pattern"),
                "has_method": elr_row.get("has_method"),
                "has_model": elr_row.get("has_model"),
                "has_specimen": elr_row.get("has_specimen"),
                "mrr": mrr,
                "mrr_grouped": mrr_grouped,
                "specificity_mismatch": (mrr_grouped > mrr) and (mrr < 1.0),
                "top1": int(candidates.iloc[0]["loinc_num"] == elr_row["loinc_num"]),
                "top3": int(
                    elr_row["loinc_num"] in candidates.head(3)["loinc_num"].values
                ),
                "top5": int(
                    elr_row["loinc_num"] in candidates.head(5)["loinc_num"].values
                ),
                **scores,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def assign_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    df = df.copy()
    df["split"] = "val"
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    _, test_idx = next(sss_test.split(df, df["loinc_num"]))
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    return df
