import pandas as pd
import re
import logging


# ---------------------------------------------------------------------------
# 1. Logger Setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2. Domain Constants
# ---------------------------------------------------------------------------
BRAND_ANCHORS = {
    "COBAS",
    "XPERT",
    "TAQPATH",
    "BIOFIRE",
    "BINAXNOW",
    "VERITOR",
    "LUMIRADX",
    "ALINITY",
    "M2000",
    "APTIMA",
    "PANTHER",
    "SOFIA",
    "QUICKVUE",
    "INTELISWAB",
    "NEUMODX",
    "ARIES",
    "LIAISON",
    "DIASORIN",
    "ATELLICA",
    "VITROS",
    "SOLANA",
    "PROCLEIX",
    "QIASTAT",
    "ALLPLEX",
    "SIMOA",
    "SHERLOCK",
    "LUCIRA",
    "ELLUME",
    "CARESTART",
    "FLOWFLEX",
    "INDICAID",
    "CLINITEST",
    "IHEALTH",
    "OSOM",
    "PANBIO",
    "ADVIA",
    "NXTAG",
    "MASSARRAY",
    "VISBY",
    "TALIS",
    "AMPLITUDE",
    "LABGUN",
    "LYRA",
    "PERKINELMER",
    "REALSTAR",
    "GENEFINDER",
    "POWERCHEK",
    "TRUPCR",
    "MICROGEM",
    "BD",
    "HEALGEN",
    "HOTGEN",
    "WANTAI",
    "FOSUN",
    "GENBODY",
    "CELLTRION",
    "SAMPINUTE",
    "IAMP",
    "BIOMEME",
    "T2SARS",
}

LOINC_SYSTEM_EXPANSION = {
    "Nph": "Nasopharynx Nasopharyngeal NP NPH Nasopharyngeal Swab",
    "Nose": "Nasal Nose Anterior Nares Nasal Swab Midturbinate AN NSL",
    "Thrt": "Throat Oropharyngeal Oropharynx OP Throat Swab",
    "Saliva": "Saliva Oral Fluid",
    "BAL": "BAL Bronchoalveolar Lavage Bronch",
    "Respiratory system specimen.upper": "Upper Respiratory Nasopharyngeal Nasal Throat NP Nose",
    "Respiratory system specimen.lower": "Lower Respiratory Bronchial Tracheal BAL LRT",  # forward compat only
    "Respiratory System Specimen": "",  # generic catch-all, always retained, no expansion needed
}

LOINC_METHOD_EXPANSION = {
    "Probe.amp.tar": "NAAT NAA PCR RT-PCR QPCR RT-QPCR MOLECULAR NUCLEIC ACID AMPLIFICATION",
    "Probe.amp.tar.CDC primer-probe set N1": "NAAT NAA PCR RT-PCR MOLECULAR",  # collapses to parent surface forms
    "Probe.amp.tar.CDC primer-probe set N2": "NAAT NAA PCR RT-PCR MOLECULAR",
    "Non-probe.amp.tar": "NAAT NAA LAMP ISOTHERMAL",
    "IA.rapid": "LATERAL FLOW AG ANTIGEN RAPID",
    "IA": "IMMUNOASSAY ELISA ECLIA AG ANTIGEN",
}


SPECIMEN_TO_LOINC_SYSTEM = {
    "NP": [
        "Nph",
        "Nasopharynx",
        "Respiratory System Specimen",
        "Respiratory system specimen.upper",
    ],
    "NASAL": [
        "Nose",
        "Respiratory system specimen.upper",
        "Respiratory System Specimen",
    ],
    "THROAT": [
        "Thrt",
        "Oropharyngeal wash",
        "Respiratory system specimen.upper",
        "Respiratory System Specimen",
    ],
    "COMBINED_NT": [
        "Nph",
        "Nose",
        "Respiratory system specimen.upper",
        "Respiratory System Specimen",
    ],
    "SALIVA": ["Saliva", "Respiratory System Specimen"],
    "BAL": ["BAL", "Respiratory System Specimen"],
    "BRONCHIAL": [
        "Respiratory system specimen.lower",
        "Bronchial",
        "Tracheal",
        "Respiratory System Specimen",
    ],
    "SPUTUM": ["Respiratory System Specimen"],
    "LRT_GENERAL": ["Respiratory System Specimen"],
    "UNKNOWN": None,  # skip filtering entirely
}


VALID_ANATOMY_MAP = {
    "Nose": ["NASAL", "UNKNOWN"],
    "Nph": ["NP", "COMBINED_NT", "UNKNOWN"],
    "Nasopharynx": ["NP", "COMBINED_NT", "UNKNOWN"],
    "Thrt": ["THROAT", "COMBINED_NT", "UNKNOWN"],
    "Saliva": ["SALIVA", "UNKNOWN"],
    "BAL": ["BAL", "LRT_GENERAL", "UNKNOWN"],
    "Bronchial": ["BAL", "TRACHEAL", "LRT_GENERAL", "UNKNOWN"],
    "Respiratory system specimen.upper": [
        "NASAL",
        "NP",
        "THROAT",
        "COMBINED_NT",
        "URT_GENERAL",
        "UNKNOWN",
    ],
    "Respiratory system specimen.lower": [
        "BAL",
        "SPUTUM",
        "TRACHEAL",
        "LRT_GENERAL",
        "UNKNOWN",
    ],
    "Respiratory System Specimen": [
        "NASAL",
        "NP",
        "THROAT",
        "BAL",
        "SPUTUM",
        "TRACHEAL",
        "SALIVA",
        "COMBINED_NT",
        "URT_GENERAL",
        "LRT_GENERAL",
        "UNKNOWN",
    ],
}


# ---------------------------------------------------------------------------
# 3. Text Cleaning & Normalization
# ---------------------------------------------------------------------------


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.upper()

    # Pass 1: Bind influenza subtypes while punctuation still present
    text = re.sub(
        r"\b(?:F|FLU|INFLUENZA)[-\s]?(?:(?:VIRUS|TYPE)[-\s]?)?A\b",
        "FLUA",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:F|FLU|INFLUENZA)[-\s]?(?:(?:VIRUS|TYPE)[-\s]?)?B\b",
        "FLUB",
        text,
        flags=re.IGNORECASE,
    )

    # Pass 2: Standardise COVID forms while punctuation still present
    text = re.sub(
        r"SARSCOV2|SARS[-\s]?COV[-\s]?2|SARS2|COVID[-\s]?19|COVID\b|CV[-\s]?19|2019[-\s]?NCOV|SARS[-\s]?CORONAVIRUS[-\s]?2",
        "SARSCOV2",
        text,
        flags=re.IGNORECASE,
    )

    # Strip non-alphanumeric after all binding is complete
    text = re.sub(r"[^A-Z0-9\s]", " ", text)

    return " ".join(text.split())


def normalize_specimen(text: str) -> str:
    """Categorizes different tokens used for a specimen into a common class"""
    if pd.isna(text) or not text.strip():
        return "UNKNOWN"
    text = text.lower()

    # Hierarchical clinical rules
    if any(
        k in text
        for k in ["combined", "combination", "combo", " and ", "/op", "/throat"]
    ):
        if any(
            k in text for k in ["nasopharyngeal", "np ", " nph", "naso", "nasopharynx"]
        ) and any(
            k in text for k in ["oropharyngeal", "opharyngeal", "throat", " op", "thrt"]
        ):
            return "COMBINED_NT"
    if any(k in text for k in ["nasopharyngeal", "np ", " nph", "nasopharynx"]):
        return "NP"
    if any(
        k in text for k in ["oropharyngeal", "opharyngeal", "throat", " op", "thrt"]
    ):
        return "THROAT"
    if any(k in text for k in ["nasal", "nares", "anterior", "turbinate", "nose"]):
        return "NASAL"
    if any(k in text for k in ["lavage", "bal", "bronchoalveolar", "bronchioalveolar"]):
        return "BAL"
    if "sputum" in text:
        return "SPUTUM"
    if any(k in text for k in ["bronchial", "tracheal", "endotracheal"]):
        return "BRONCHIAL"
    if "saliva" in text:
        return "SALIVA"
    if "lower" in text or "lrt" in text:
        return "LRT_GENERAL"
    if "upper" in text or "urt" in text:
        return "URT_GENERAL"

    logger.warning(f"Unrecognized specimen text: '{text}'. Mapping to UNKNOWN.")
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# 3. Quality testing/validation
# ---------------------------------------------------------------------------


warned_systems = set()


def is_anatomically_valid(loinc_sys, spec_norm):
    """
    Checks if a pair is valid. Warns once per unknown system.
    """
    allowed_list = VALID_ANATOMY_MAP.get(loinc_sys)

    if allowed_list is None:
        if loinc_sys not in warned_systems:
            logger.warning(
                f"UNENCOUNTERED SYSTEM: '{loinc_sys}' not in VALID_ANATOMY_MAP. "
                "Defaulting to 'Permissive' (True). Please audit this system."
            )
            warned_systems.add(loinc_sys)
        return True
    return spec_norm in allowed_list


def filter_clinical_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Active Filter: Drops rows where system does not belong to the specimen_norm anatomy."""
    mask = [
        is_anatomically_valid(sys, norm)
        for sys, norm in zip(df["System"], df["specimen_norm"])
    ]
    return df[mask].copy()


def test_anatomical_integrity(df):
    """
    Audits the simulated dataset against the master VALID_ANATOMY_MAP.
    Identifies rows where the System/Specimen pairing is medically impossible.
    """
    # Identify violations using our shared logic
    invalid_mask = [
        not is_anatomically_valid(sys, norm)
        for sys, norm in zip(df["System"], df["specimen_norm"])
    ]

    violations = df[invalid_mask]

    # Reporting
    if not violations.empty:
        logger.warning(
            f"ANATOMY VIOLATION: {len(violations)} rows found with invalid pairings. "
            f"Impacted Systems: {violations['System'].unique().tolist()}"
        )

        # Provide samples of failures
        logger.debug(
            f"Violation Samples:\n{violations[['System', 'specimen_norm']].head(10)}"
        )
    else:
        logger.info("Anatomical Integrity Passed: All pairs align with Clinical Map.")


# ---------------------------------------------------------------------------
# 4. Utilities
# ---------------------------------------------------------------------------


def filter_corpus_by_specimen(corpus_df, specimen_norm):
    if specimen_norm == "UNKNOWN":
        return corpus_df
    valid_systems = SPECIMEN_TO_LOINC_SYSTEM.get(specimen_norm, [])
    return corpus_df[
        (corpus_df["System"] == "Respiratory System Specimen")
        | (corpus_df["System"].isin(valid_systems))
    ]
