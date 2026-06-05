# Preprocesses LIVD and LOINC files to produce Covid-19 Surveillance LOINC codes and a metadata-enriched LIVD file for ELR simulation.
# Also produces distractor LOINCs for TF-IDF corpus construction.

import pandas as pd
import re
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("logs/loinclivdpreprocessing.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "LOINC_NUM",
    "COMPONENT",
    "PROPERTY",
    "TIME_ASPCT",
    "SYSTEM",
    "SCALE_TYP",
    "METHOD_TYP",
    "CLASS",
    "STATUS",
    "LONG_COMMON_NAME",
    "RELATEDNAMES2",
    "PanelType",
]

# Specimen exclusions, immunology testing exclusion (to limit scope of project)
SPECIMEN_EXCLUSIONS = r"\b(?:Bld|Ser|Plas|Stool|Isolate|XXX|Tiss|Exhl gas|Asp|Cornea)\b"
OMIT_IMMUNOLOGY = r"\b(?:IgG|IgM|IgA|Ab|Antibody|Antibodies)\b"

# Identify antigen vs naat testing
ANTIGEN_REGEX = (
    r"\b(?:IF|IA|EIA|ELISA|FIA|CLIA|LFIA|rapid|antigen|immunofluorescence)\b|\bag\b"
)
NAAT_REGEX = r"\b(?:NAAT|NAA|PCR|RT[- ]?PCR|TMA|LAMP|SDA)\b|nucleic[- ]acid|amplif"

# Pathogens present in LIVD device submissions and simulation scope —
# used to select clinically realistic distractor codes.
DISTRACTOR_PATHOGENS = [
    "Influenza",
    "Respiratory syncytial",
    "Metapneumovirus",
    "Parainfluenza",
    "Adenovirus",
    "Rhinovirus",
    "Enterovirus",
    "Bocavirus",
    "Bordetella",
    "Mycoplasma",
    "Chlamydophila",
    "Respiratory",
    "MERS",
]

# Components that are out-of-scope even if a respiratory pathogen matches
DISTRACTOR_EXCLUSIONS = [
    "Pneumocystis",
    "Varicella",
    "Cytomegalovirus",
    "Deprecated",
]


# ---------------------------------------------------------------------------
# Outputs 1 + 2: LOINC reference and enriched LIVD
# ---------------------------------------------------------------------------


def preprocess_clinical_assets(loinc_path: str, livd_path: str) -> pd.DataFrame:
    """
    Produces two outputs:
      data/processed/covid_surveillance_loinc.csv  — filtered LOINC reference
      data/processed/livd_for_simulation.csv       — LIVD enriched with LOINC metadata

    Returns the filtered LOINC DataFrame for downstream use (e.g. distractor building).
    """
    # 1. Load data
    loinc = pd.read_csv(loinc_path, low_memory=False)
    livd = pd.read_csv(livd_path)

    # 2. Define universe: union of performed and ordered LOINC codes in LIVD
    livd_codes = (
        pd.concat([livd["Test Performed LOINC Code"], livd["Test Ordered LOINC Code"]])
        .dropna()
        .unique()
    )

    # 3. Filter LOINC to universe, then isolate active viral testing
    df = loinc[loinc["LOINC_NUM"].isin(livd_codes)].copy()
    df = df[df["METHOD_TYP"] != "Sequencing"]
    df = df[~df["SYSTEM"].str.contains(SPECIMEN_EXCLUSIONS, case=False, na=False)]
    df = df[~df["COMPONENT"].str.contains(OMIT_IMMUNOLOGY, case=False, na=False)]

    # 4. Coerce key string columns before regex masks
    df["METHOD_TYP"] = df["METHOD_TYP"].astype("string").fillna("")
    df["COMPONENT"] = df["COMPONENT"].astype("string").fillna("")
    df["LONG_COMMON_NAME"] = df["LONG_COMMON_NAME"].astype("string").fillna("")
    df = df[REQUIRED_COLUMNS]

    # 5. Classify method
    def contains(col, pattern):
        return df[col].str.contains(pattern, case=False, na=False)

    antigen_mask = (
        contains("METHOD_TYP", ANTIGEN_REGEX)
        | contains("COMPONENT", ANTIGEN_REGEX)
        | contains("LONG_COMMON_NAME", ANTIGEN_REGEX)
    )
    naat_mask = (
        contains("METHOD_TYP", NAAT_REGEX)
        | contains("COMPONENT", NAAT_REGEX)
        | contains("LONG_COMMON_NAME", NAAT_REGEX)
    )

    df["method_class"] = "unknown"
    df.loc[antigen_mask & ~naat_mask, "method_class"] = "antigen"
    df.loc[naat_mask & ~antigen_mask, "method_class"] = "naat"
    df.loc[antigen_mask & naat_mask, "method_class"] = "mixed/panel"

    # 6. Standardize column names for downstream use
    df.columns = [col.lower() for col in df.columns]

    # Output 1: cleaned LOINC reference
    df.to_csv("data/processed/covid_surveillance_loinc.csv", index=False)
    print(f"[1/3] LOINC reference: {df.shape[0]} rows → covid_surveillance_loinc.csv")

    # Output 2: LIVD enriched with LOINC metadata (inner join keeps only validated codes)
    sim_df = livd.merge(
        df, left_on="Test Performed LOINC Code", right_on="loinc_num", how="inner"
    )
    sim_df.to_csv("data/processed/livd_for_simulation.csv", index=False)
    print(
        f"[2/3] Simulation source: {livd.shape[0]} LIVD rows → {sim_df.shape[0]} after merge"
        f" → livd_for_simulation.csv"
    )

    return df


# ---------------------------------------------------------------------------
# Output 3: Distractor LOINC codes
# ---------------------------------------------------------------------------


def build_distractor_loincs(loinc_full_path: str, df_loinc_eval: pd.DataFrame) -> None:
    """
    Selects non-COVID respiratory LOINC codes to serve as retrieval distractors.
    Codes must share method_typ and system with at least one eval code so they
    represent realistic confusables. COVID codes are excluded as they're in the main covid loincs file.
    Produces:
      data/processed/distractor_loincs.csv
    """
    df_full = pd.read_csv(loinc_full_path, low_memory=False)
    df_full.columns = df_full.columns.str.lower()

    eval_loincs = set(df_loinc_eval["loinc_num"].values)
    eval_method_types = set(df_loinc_eval["method_typ"].dropna().unique())
    eval_systems = set(df_loinc_eval["system"].dropna().unique())

    # Candidates: not an eval target, shares method_typ and system
    candidates = df_full[
        ~df_full["loinc_num"].isin(eval_loincs)
        & df_full["method_typ"].isin(eval_method_types)
        & df_full["system"].isin(eval_systems)
    ].copy()

    # Keep only respiratory/relevant-pathogen codes
    pathogen_pattern = "|".join(DISTRACTOR_PATHOGENS)
    candidates = candidates[
        candidates["component"].str.contains(
            pathogen_pattern, case=False, na=False, regex=True
        )
        | candidates["long_common_name"].str.contains(
            pathogen_pattern, case=False, na=False, regex=True
        )
    ]

    # Drop out-of-scope components
    exclusion_pattern = "|".join(DISTRACTOR_EXCLUSIONS)
    candidates = candidates[
        ~candidates["long_common_name"].str.contains(
            exclusion_pattern, case=False, na=False, regex=True
        )
    ]

    # Drop COVID-adjacent codes — these share SARSCOV2 tokens and don't raise IDF
    candidates = candidates[
        ~candidates["long_common_name"].str.contains(
            r"SARS|COVID|coronavirus", case=False, na=False, regex=True
        )
    ]

    candidates.to_csv("data/processed/distractor_loincs.csv", index=False)
    print(
        f"[3/3] Distractors: {candidates['loinc_num'].nunique()} unique codes"
        f" → distractor_loincs.csv"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    LOINC_PATH = "data/raw/Loinc.csv"
    LIVD_PATH = "data/raw/LIVD-SARS-CoV-2.csv"

    df_eval = preprocess_clinical_assets(LOINC_PATH, LIVD_PATH)
    build_distractor_loincs(LOINC_PATH, df_eval)

    print("\nPreprocessing complete. Outputs written to data/processed/.")

# copyright double check
# covid_loinc = pd.read_csv("data/processed/covid_surveillance_loinc.csv")
# loinc = pd.read_csv("data/raw/Loinc.csv")
# loinc.loc[loinc.LOINC_NUM.isin(covid_loinc.loinc_num.unique()),"EXTERNAL_COPYRIGHT_NOTICE"].unique()
