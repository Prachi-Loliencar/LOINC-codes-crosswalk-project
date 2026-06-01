import pandas as pd
import numpy as np
import random
import re

import logging

from src.clinical_utils import (
    BRAND_ANCHORS,
    clean_text,
    normalize_specimen,
    filter_clinical_alignment,
    test_anatomical_integrity,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("logs/simulation.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token sets for model name simplification (O(1) lookups)
# ---------------------------------------------------------------------------

KEEP_MODIFIERS = {
    "LIAT",
    "XPRESS",
    "OMNI",
    "FUSION",
    "SPOTFIRE",
    "PLUS",
    "6800",
    "8800",
    "M",
    "MAX",
    "VERITOR",
    "2",
    "ONE",
    "CRISPR",
    "CENTAUR",
    "DX",
}


MODEL_DROP = {
    "COVID",
    "COVID19",
    "SARS",
    "COV 2",
    "COV",
    "COV2",
    "PCR",
    "RT",
    "QPCR",
    "AG",
    "ANTIGEN",
    "TEST",
    "ASSAY",
    "KIT",
    "PANEL",
    "SYSTEM",
    "REAL",
    "TIME",
    "NUCLEIC",
    "ACID",
    "FOR",
    "THE",
    "ON",
    "AUTHORIZED",
    "HHS",
    "OASH",
    "RNA",
    "RESPIRATORY",
    "DETECTION",
    "FLUORESCENT",
}

# Pre-compiled Regexes for fast matching
ANALYTE_RE = re.compile(
    r"\b("
    # SARS-CoV-2 surface forms
    r"SARSCOV2|SARS[\s-]?COV[\s-]?2|SARS2|SC2|"
    r"COVID[\s-]?19|COVID|COV2|CV19|"
    r"2019[\s-]?NCOV|CORONAVIRUS|"
    r"SARS[\s-]RELATED|SARS[\s-]COV|SARS|"
    r"SARBECOVIRUS|SARSREL|"
    # Influenza — post clean_text these are already bound to FLUA/FLUB
    r"FLUA|FLUB|"
    # RSV
    r"RSV|"
    # Other respiratory pathogens
    r"HMPV|MPV|METAPNEUMOVIRUS|"
    r"PIV|PARAINFLUENZA|"
    r"ADV|ADENOVIRUS|"
    r"RV|RHINOVIRUS|ENTEROVIRUS|"
    r"HCOV|BOCAVIRUS|HBOV|"
    r"BPERT|PERTUSSIS|BORDETELLA|"
    r"MYCOPLASMA|MYPNEU|"
    r"CHLAMYDOPHILA|CPNEU|"
    # Gene targets — present in ELR strings for some COVID codes
    r"NGENE|RDRP|ORF1AB|SGENE|EGENE|MGENE|NSP2"
    r")\b",
    re.I,
)

METHOD_RE = re.compile(
    r"\b(PCR|RT[- ]?PCR|QPCR|NAAT|NAA|NAT|AMP(LIFIED)?|LAMP|TMA|AG|ANTIGEN|RAPID)\b",
    re.I,
)

SPEC_RE = re.compile(
    r"\b(NP|N\.P\.|NPS|NASOPH(ARYNGEAL)?|NASAL|ANTERIOR\s+NARES|AN\b|MID\s+TURBINATE|MT\b|THROAT|OP\b|OROPH(ARYNGEAL)?|SALIVA|ORAL\s+FLUID|SPUTUM|BAL|BRONCH(OALVEOLAR)?|TRACHE(AL)?|ENDOTRACHEAL|ET\s+ASP)\b",
    re.I,
)

INTERP_RE = re.compile(
    r"\b(REPORT|FINAL|RES(ULT)?|INTERP(RETATION)?|STATUS|SUMMARY)\b", re.I
)

# ---------------------------------------------------------------------------
# Configuration & Surface Maps
# ---------------------------------------------------------------------------
MIN_SEEDS = 3
COV_ABBRV_PROB = 0.5
TARGET_DELETION_PROB = 0.3
APPEND_PROB = 0.1
SPECIMEN_ABBRV_PROB = 0.7
METHOD_SHORT_PROB = 0.6
TYPO_PROBABILITY = 0.10

ANALYTE_TARGETS = ["RNA", "nucleocapsid", "protein", "antigen", "FLUA", "FLUB"]
ANALYTE_SURFACE_MAP = {
    "SARSCOV2": ["SARS-CoV-2", "COVID-19", "COVID19", "CV19", "CORONAVIRUS", "SARS2"],
    "FLUA": ["FLU A", "INFLUENZA A", "F-A", "FLUA"],  # Added to distinguish
    "FLUB": ["FLU B", "INFLUENZA B", "F-B", "FLUB"],  # Added to distinguish
    "nucleocapsid": ["N-GENE", "N-PROT", "N-AG", "N-ANTIGEN", "N-PROTEIN", ""],
    "protein": ["PROT", "AG", "ANTIGEN", "IMMUNO", ""],
    "antigen": ["AG", "ANTIGEN", "IMM", ""],
    "RNA": ["NAA", "PCR", "AMP", "NUCLEIC", "GENETIC", "DETECTION", ""],
    "appends": [
        "STATUS",
        "INTERP",
        "RES",
        "RESULT",
        "FINDING",
        "FINAL",
        "RPT",
        "FIN",
        "SUMMARY",
        "LAB RESULT",
    ],
}

SPECIMEN_SURFACE_MAP = {
    "NP": [
        "nasopharyngeal swab",
        "np swab",
        "nasopharyngeal aspirate",
        "nasopharyngeal wash",
        "nph",
    ],
    "NASAL": [
        "anterior nasal swab",
        "nasal swab",
        "mid turbinate nasal swab",
        "anterior nares",
        "midturbinate",
        "nasal wash",
        "nasal aspirate",
    ],
    "THROAT": ["oropharyngeal swab", "throat swab", "opharyngeal swab", "thrt"],
    "COMBINED_NT": [
        "nasopharyngeal and oropharyngeal swab",
        "combined np op swab",
        "nasal and pharyngeal swab combination",
        "nasopharyngeal oropharyngeal swab",
        "nasopharyngeal aspirate oropharyngeal swab",
    ],
    "SALIVA": ["saliva", "oral fluid"],
    "SPUTUM": ["sputum", "induced sputum", "expectorated sputum"],
    "BAL": [
        "bronchoalveolar lavage",
        "bal",
        "bronchioalveolar lavage",
        "broncho alveolar lavage fluid",
    ],
    "BRONCHIAL": ["bronchial aspirate", "tracheal aspirate", "endotracheal aspirates"],
    "URT_GENERAL": [
        "upper respiratory",
        "upper resp",
        "respiratory system specimen.upper",
    ],
    "LRT_GENERAL": ["lower respiratory tract aspirates", "lower resp", "lrt"],
    "UNKNOWN": ["", "respiratory system specimen", "swab"],
}


METHOD_SURFACE_MAP = {
    # Core probe amplification — includes CDC N1/N2 primer-probe set variants
    "probe.amp.tar": ["PCR", "RT-PCR", "NAAT", "NAA", "QPCR", "RT-QPCR"],
    "probe.amp.tar.cdc primer-probe set n1": ["PCR", "RT-PCR", "NAAT", "NAA"],
    "probe.amp.tar.cdc primer-probe set n2": ["PCR", "RT-PCR", "NAAT", "NAA"],
    "probe.amp.tar.primer-probe set n1": ["PCR", "RT-PCR", "NAAT", "NAA"],
    # Non-probe amplification (LAMP, isothermal, etc.)
    "non-probe.amp.tar": ["NAAT", "NAA", "LAMP", "ISOTHERMAL"],
    # Immunoassay variants
    "ia.rapid": ["RAPID AG", "RAPID ANTIGEN", "RAPID"],
    "ia": ["ANTIGEN", "AG", "ELISA", "IMMUNOASSAY"],
    # Immunofluorescence
    "if": ["IF"],
}

MANUFACTURER_SHORTHAND = {
    "Abbott Diagnostics Scarborough": ["ABBOTT DX", "ABBOTT SCARBOROUGH", "ABT"],
    "Becton Dickinson and Company (BD)": ["BD", "BECTON DICKINSON"],
    "Laboratory Corporation of America (Labcorp)": ["LABCORP", "LCA"],
    "Quest Diagnostics": ["QUEST", "QUEST DX"],
    "Thermo Fisher Scientific": ["THERMO", "FISHER", "TFS", "TMO"],
    "Roche Molecular Systems": ["ROCHE", "ROCHE MOLECULAR"],
}

SIGNAL_TOKENS = {
    "SARS",
    "COV",
    "COV2",
    "COVID",
    "CORONAVIRUS",
    "SARSCOV2",
    "RNA",
    "NAA",
    "NAAT",
    "PCR",
    "NUCLEIC",
    "GENETIC",
    "AMP",
    "ANTIGEN",
    "AG",
    "PROTEIN",
    "PROT",
    "NUCLEOCAPSID",
    "NP",
    "NASAL",
    "NASOPHARYNGEAL",
    "THROAT",
    "OROPHARYNGEAL",
    "SALIVA",
    "SPUTUM",
    "BAL",
    "RESPIRATORY",
}

EXPORT_COLUMNS = [
    "loinc_num",
    "long_common_name",
    "component",
    "system",
    "method_typ",
    "scale_typ",
    "specimen_norm",
    "specimen_clean",
    "System",
    "Model",
    "method_class",
    "method_clean",
    "analyte_clean",
    "elr_name",
    "has_specimen",
    "has_method",
    "has_model",
    "noise_corruption",
    "noise_compression",
    "noise_omission",
    "noise_total",
    "noise_level",
    "analyte_len",
    "info_score",
    "coverage_pattern",
    "has_analyte",
    "seed_id",
    "variant_id",
]


# Templates for component inclusion and their probabilities
TEMPLATE_CONFIG = {
    "a": {"has_mod": 0, "has_meth": 1, "has_spec": 1},
    "b": {"has_mod": 0, "has_meth": 1, "has_spec": 0},
    "c": {"has_mod": 1, "has_meth": 1, "has_spec": 0},
    "d": {"has_mod": 0, "has_meth": 0, "has_spec": 1},
    "e": {"has_mod": 0, "has_meth": 0, "has_spec": 0},
    "f": {"has_mod": 1, "has_meth": 1, "has_spec": 1},
    "g": {"has_mod": 1, "has_meth": 0, "has_spec": 0},
}
TEMPLATE_WEIGHTS = [0.27, 0.27, 0.13, 0.09, 0.07, 0.09, 0.08]


COMPONENT_WEIGHTS = {"has_mod": 0, "has_meth": 1, "has_spec": 0}
FULL_COMPONENTS = set(COMPONENT_WEIGHTS.keys())


# only absense of method is considered noise out of the components since specimen
# is not highly discriminative in this dataset
def template_structural_noise(config: dict) -> int:
    return sum(COMPONENT_WEIGHTS[k] for k in FULL_COMPONENTS if not config[k])


def make_noise_tracker() -> dict:
    return {"corruption": 0, "compression": 0, "omission": 0}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def bind_pathogen_letters(text: str) -> str:
    """
    Binds 'A' or 'B' to Flu/Influenza keywords to preserve signal.
    Example: 'Flu A' -> 'FLUA', 'Influenza B' -> 'INFLUENZAB'
    """
    if not isinstance(text, str):
        return ""

    # 1. Handle 'Flu A/B' and 'Influenza A/B'
    text = re.sub(
        r"\b(flu|influenza|inf|f)\s+([ab])\b", r"\1\2", text, flags=re.IGNORECASE
    )

    # 2. Handle 'A & B' or 'A and B' appearing in multiplex kit names
    # This turns 'Flu A and B' into 'FLUA FLUB'
    text = re.sub(
        r"\b(flu|influenza)\s+a\s*(?:and|&|/)\s*b\b",
        r"\1a \1b",
        text,
        flags=re.IGNORECASE,
    )
    return text


def analyte_clean_preserve(text: str) -> str:
    """
    Semantic wrapper for Vendor Analyte Name.
    Currently uses the global engine, but preserved as a unique
    entry point for domain-specific analyte rules for future projects.
    """
    if pd.isna(text):
        return ""
    return clean_text(str(text))


def simplify_model_name(model: str) -> str:
    # remove component information and noise from model name and simplify
    if pd.isna(model) or not str(model).strip():
        return ""

    s = re.sub("SARS-COV-2", " ", str(model).upper())
    s = re.sub(r"\([^)]*\)|[^A-Z0-9 ]|\bLDT\b", " ", s)
    tokens = [t for t in re.sub(r"\s+", " ", s).strip().split() if t not in MODEL_DROP]

    if not tokens:
        return ""
    anchor = next((t for t in tokens if t in BRAND_ANCHORS), tokens[0])
    modifier = next((t for t in tokens if t in KEEP_MODIFIERS and t != anchor), None)
    return f"{anchor} {modifier}".strip() if modifier else anchor


def replace_unknown(text: str) -> str:
    text = re.sub(r"\r", "", text, flags=re.IGNORECASE)
    matches = re.findall(r"\((.*?)\)", text)
    return matches[0].split("^")[1] if matches and "^" in matches[0] else text


def create_clean_seeds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def split_specimens(x):
        """Splits by newline but preserves context for normalization."""
        return [
            line.strip()
            for line in str(x).split("\n")
            if line.strip() and line.strip().lower() != "or"
        ]

    # 1. Track the source
    df["parent_row_id"] = df.index

    # 2. Split by newline but keep context (parentheses) for normalization
    df["raw_lines"] = df["Vendor Specimen Description"].apply(split_specimens)
    df = df.explode("raw_lines").reset_index(drop=True)

    # 3. Normalize ground truth using the raw line (with context)
    df["specimen_norm"] = df["raw_lines"].apply(normalize_specimen)

    # 4. Strip parentheses and dangling conjunctions
    df["specimen_clean"] = (
        df["raw_lines"]
        .str.replace(r"\(.*?\)", "", regex=True)  # Remove (SNOMED codes)
        .str.replace(
            r"\b(or|and)\b\s*$", "", regex=True, flags=re.IGNORECASE
        )  # Remove trailing 'or/and'
        .str.strip()
    )

    # 5. Fallback for remaining UNKNOWNs
    if "replace_unknown" in globals():  # Check if helper exists
        mask = df["specimen_norm"] == "UNKNOWN"
        df.loc[mask, "specimen_norm"] = df.loc[mask, "raw_lines"].apply(
            lambda x: normalize_specimen(replace_unknown(x))
        )

    df["analyte_clean"] = df["Vendor Analyte Name"].apply(analyte_clean_preserve)
    df["model_clean"] = df["Model"].apply(clean_text)

    # Coalesce method logic
    df["method_clean"] = df["Method"].fillna(df.get("method_class", ""))

    num_before = df.shape[0]
    df = df[~df["specimen_norm"].str.lower().isin(["or"])].copy()
    logger.info(
        f"Processed {num_before} rows. Dropped {num_before - df.shape[0]} rows due to '\\or' or invalid artifacts."
    )
    return df


# artifact corrected in this function was only in a single seed
def clean_vendor_specimen(text: str) -> str:
    if pd.isna(text):
        return text
    return re.sub(
        r"\basopharyngeal\b", "nasopharyngeal", str(text), flags=re.IGNORECASE
    )


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------


def analyte_perturb(text: str, noise: dict) -> str:
    """Perturb analyte name and record noise by category with clinical guardrails."""
    # 1. Clean formal LOINC noise
    text = re.sub(r"\b(INTERPRETATION|RESULT)S?\b", "", text, flags=re.IGNORECASE)

    # 2. Canonicalization of the Core Analyte
    # We do this first so further perturbations act on the standardized form
    if "SARSCOV2" in text.upper():
        if random.random() < COV_ABBRV_PROB:
            replacement = random.choice(ANALYTE_SURFACE_MAP["SARSCOV2"])
            noise["compression"] += 1
        else:
            replacement = "SARS-CoV-2"
        text = re.sub("SARSCOV2", replacement, text, flags=re.IGNORECASE)

    # 3. Targeted Perturbations (RNA, Proteins, etc.)
    if random.random() < TARGET_DELETION_PROB:
        # Sort by length descending to avoid partial matches (e.g., 'antigen' before 'antigenic')
        for target in sorted(ANALYTE_TARGETS, key=len, reverse=True):
            pattern = rf"\b{target}\b"
            if re.search(pattern, text, flags=re.IGNORECASE):
                replacement = random.choice(ANALYTE_SURFACE_MAP.get(target, [""]))
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)

                if not replacement.strip():
                    noise["omission"] += 1
                else:
                    noise["compression"] += 1
                break  # Only perturb one target per call to maintain signal

    # 4. Appendix Noise (domain-recoverable surface addition, not character corruption)
    if random.random() < APPEND_PROB:
        append_word = random.choice(ANALYTE_SURFACE_MAP.get("appends", []))
        text = f"{text} {append_word}"
        noise["compression"] += 1

    # 5. Final Sanitization
    # Remove double spaces and strip
    clean_result = re.sub(r"\s+", " ", text).strip().upper()

    return clean_result


def add_typos(text: str, noise: dict, prob: float = 0.10, decay: float = 0.1) -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # Optimized string building
    text_list = list(text)

    while random.random() < prob and text_list:
        typo_type = random.choice(["swap", "skip", "extra"])
        idx = random.randint(0, len(text_list) - 1)
        if typo_type == "swap" and idx < len(text_list) - 1:
            # Only swap if we aren't moving a space into a word
            if text_list[idx] != " " and text_list[idx + 1] != " ":
                text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
        elif typo_type == "skip":
            text_list.pop(idx)
        elif typo_type == "extra":
            text_list.insert(
                idx, text_list[idx] if random.random() < 0.5 else random.choice(chars)
            )

        noise["corruption"] += 1
        prob *= decay

    return "".join(text_list)


def manufacturer_perturb(text: str) -> str:
    if text in MANUFACTURER_SHORTHAND:
        return random.choice(MANUFACTURER_SHORTHAND[text])
    text = re.sub(r",? (Inc|LLC|Ltd|Corp|GmbH|Co|Ltd|UK Ltd|USA Inc)", "", str(text))
    text = re.sub(
        r" (Laboratories|Diagnostics|Scientific|Bioscience|Biotech)", "", text
    )
    if "(" in text:
        text = (
            re.search(r"\((.*?)\)", text).group(1)
            if random.random() < 0.8
            else re.sub(r"\(.*?\)", "", text)
        )
    return clean_text(text).upper()


# ---------------------------------------------------------------------------
# Seed factory
# ---------------------------------------------------------------------------
def name_seed_factory(row: pd.Series, seed: int = 33) -> pd.Series:
    random.seed(seed)
    noise = make_noise_tracker()

    template = random.choices(
        list(TEMPLATE_CONFIG.keys()), weights=TEMPLATE_WEIGHTS, k=1
    )[0]
    config = TEMPLATE_CONFIG[template]

    has_mod, has_meth, has_spec = (
        config["has_mod"],
        config["has_meth"],
        config["has_spec"],
    )
    noise["omission"] += template_structural_noise(config)

    analyte = analyte_perturb(row["analyte_clean"], noise)
    if len(analyte) <= 3:
        noise["omission"] += 2  # analyte effectively destroyed

    model = simplify_model_name(row["Model"])
    if not model:
        has_mod = 0

    specimen = ""
    if has_spec:
        specimen = random.choice(SPECIMEN_SURFACE_MAP[row["specimen_norm"]])
        if row["specimen_norm"] in ["NP", "NASAL", "THROAT"] and random.random() < 0.4:
            specimen += " SWAB"

    method = ""
    if has_meth:
        method_key = row["method_clean"].lower()
        method_options = METHOD_SURFACE_MAP.get(method_key)
        if not method_options:
            logger.warning(
                f"No METHOD_SURFACE_MAP entry for: '{method_key}' — skipping method component"
            )
            has_meth = 0
            method = ""
        else:
            method = random.choice(method_options)

    parts = [
        p
        for p, flag in [
            (model, has_mod),
            (analyte, True),
            (method, has_meth),
            (specimen, has_spec),
        ]
        if flag and p
    ]
    raw_string = random.choice([" ", " / ", " | ", " - "]).join(parts).upper()
    final_string = add_typos(raw_string, noise, prob=TYPO_PROBABILITY)

    noise_total = sum(noise.values())
    noise_level = (
        "low" if noise_total <= 1 else "medium" if noise_total <= 3 else "high"
    )

    return pd.Series(
        {
            "elr_name": final_string,
            "has_specimen": has_spec,
            "has_method": has_meth,
            "has_model": has_mod,
            "noise_corruption": noise["corruption"],
            "noise_compression": noise["compression"],
            "noise_omission": noise["omission"],
            "noise_total": noise_total,
            "noise_level": noise_level,
        }
    )


# ---------------------------------------------------------------------------
# Dataset simulation
# ---------------------------------------------------------------------------
def simulate_elr_dataset(df_clean: pd.DataFrame, n_variants: int = 5) -> pd.DataFrame:
    if "elr_name" in df_clean.columns:
        raise Exception("ELR data already present.")

    df_clean = df_clean.copy()
    df_clean["seed_id"] = range(len(df_clean))
    df_expanded = (
        pd.concat([df_clean] * n_variants).sort_values("seed_id").reset_index(drop=True)
    )
    df_expanded["variant_id"] = df_expanded.groupby("seed_id").cumcount()

    df_seeds = df_expanded.apply(
        lambda row: name_seed_factory(
            row, seed=int(row["seed_id"] * 1000 + row["variant_id"])
        ),
        axis=1,
    )
    return pd.concat([df_expanded, df_seeds], axis=1)


# ---------------------------------------------------------------------------
# Unit testing and fitness check
# ---------------------------------------------------------------------------


def score_elr_info(elr: str):
    """
    Returns (features dict) for an ELR string:
      - has_analyte, has_method, has_specimen, has_interp, has_platform
      - coverage_pattern (e.g., 'A+M+S')
      - info_score (weighted)
    """

    s = "" if pd.isna(elr) else str(elr)
    s = clean_text(s)
    s = re.sub(r"\s+", " ", s).strip()

    has_analyte = bool(ANALYTE_RE.search(s))
    has_method = bool(METHOD_RE.search(s))
    has_specimen = bool(SPEC_RE.search(s))
    has_interp = bool(INTERP_RE.search(s))
    # has_platform = bool(platform_re.search(s)) if platform_re else False

    # Weighted score: method/specimen more discriminative for LOINC
    info_score = (
        1 * has_analyte
        + 2 * has_method
        + 0 * has_specimen
        + 0 * has_interp  # noise token, no retrieval signal
        # + 1 * has_platform    # useful for brand imputation
    )

    parts = []
    if has_analyte:
        parts.append("A")
    if has_method:
        parts.append("M")
    if has_specimen:
        parts.append("S")
    if has_interp:
        parts.append("I")
    # if has_platform:
    #   parts.append("P")
    coverage_pattern = "+".join(parts) if parts else "NONE"

    return {
        "has_analyte": has_analyte,
        "has_method": has_method,
        "has_specimen": has_specimen,
        "has_interp": has_interp,
        # "has_platform": has_platform,
        "coverage_pattern": coverage_pattern,
        "info_score": info_score,
    }


def test_vocabulary_coverage(df):
    required_norms = {"NASAL", "NP", "THROAT", "BAL", "SPUTUM", "SALIVA"}
    missing = required_norms - set(df["specimen_norm"].unique())
    if missing:
        logger.warning(f"Warning: Missing specimen anchors for: {missing}")
    else:
        logger.info("Vocabulary Coverage Passed.")


def test_key_uniqueness(df):
    if df.set_index(["test_key", "specimen_norm"]).index.is_unique:
        logger.info(
            f"Row Uniqueness Passed: {df.shape[0]} distinct clinical configurations."
        )
    else:
        logger.warning(
            "Duplicate Error: Some seeds are identical across Key and Specimen."
        )


# ---------------------------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Use the enriched file that already has LOINC metadata merged
    livd_enriched = pd.read_csv("data/processed/livd_for_simulation.csv")

    df_seeds = create_clean_seeds(livd_enriched)
    df_seeds = filter_clinical_alignment(df_seeds)

    # drop invalid rows
    df_seeds = df_seeds[df_seeds["method_clean"] != ""].copy()
    df_seeds = df_seeds[
        ~df_seeds["Component"].str.lower().str.contains("internal control", na=False)
    ].copy()

    df_seeds["test_key"] = (
        df_seeds["Component"]
        + "|"
        + df_seeds["method_typ"]
        + "|"
        + df_seeds["System"]
        + "|"
        + df_seeds["Vendor Analyte Name"].str.upper().str.strip()
    )

    # drop duplicate seeds
    df_seeds_deduped = df_seeds.drop_duplicates(
        subset=["test_key", "specimen_norm"]
    ).reset_index(drop=True)
    logger.info(
        f"Shape or original clean seeds data: {df_seeds.shape}; after deduplicating: {df_seeds_deduped.shape}"
    )

    support = df_seeds_deduped.groupby(["loinc_num", "specimen_norm"]).size()
    low_support = support[support < MIN_SEEDS]
    if not low_support.empty:
        logger.info(
            f"Found {len(low_support)} LOINC/Specimen combinations with fewer than {MIN_SEEDS} seeds. "
            "Perturbation diversity for these codes will be limited."
        )
    logger.debug(f"Low support details: \n{low_support}")

    # Dropping low support seeds
    bef = df_seeds_deduped.shape[0]
    loinc_counts = df_seeds_deduped.loinc_num.value_counts()
    keep_loincs = loinc_counts[loinc_counts >= MIN_SEEDS].index
    df_seeds_deduped = df_seeds_deduped[df_seeds_deduped.loinc_num.isin(keep_loincs)]
    aft = df_seeds_deduped.shape[0]
    num_loincs = sum(loinc_counts < MIN_SEEDS)
    logger.info(
        f"Dropped {bef - aft} seeds due to fewer than 3 seeds across {num_loincs} LOINC codes."
    )

    logger.info("Starting unit tests:")
    test_anatomical_integrity(df_seeds_deduped)
    test_vocabulary_coverage(df_seeds_deduped)
    test_key_uniqueness(df_seeds_deduped)

    elr_dataset = simulate_elr_dataset(df_seeds_deduped, 12)
    logger.info(f"Simulated {elr_dataset.shape[0]} seeds.")

    df_seeds_deduped.to_csv("data/processed/clean_seeds.csv", index=False)

    # calculate word count
    elr_dataset["analyte_len"] = elr_dataset["elr_name"].str.split().str.len()

    scores_list = [score_elr_info(name) for name in elr_dataset["elr_name"]]
    scores_df = pd.DataFrame(scores_list)

    elr_dataset = pd.concat([elr_dataset, scores_df], axis=1)

    df_export = elr_dataset[EXPORT_COLUMNS].copy()
    df_export.to_csv("data/processed/elr_simulated.csv", index=False)
