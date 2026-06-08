# tests/test_retrieval.py

import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, ".")

from src.clinical_utils import clean_text, normalize_specimen
from src.model_building_utils import normalize_elr, expand_loinc_lcn


# ---------------------------------------------------------------------------
# 1. Pure transformation tests — fast, no I/O, deterministic
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_covid_aliases_canonicalized(self):
        for variant in ["COVID-19", "COVID19", "covid-19", "SARS-CoV-2", "sars-cov-2"]:
            assert clean_text(variant) == "SARSCOV2", f"Failed for: {variant}"

    def test_flu_binding(self):
        assert clean_text("Flu A") == "FLUA"
        assert clean_text("Influenza A") == "FLUA"
        assert clean_text("Influenza B") == "FLUB"

    def test_hyphen_stripped(self):
        assert "-" not in clean_text("RT-PCR")

    def test_empty_string(self):
        # Should not raise
        result = clean_text("")
        assert isinstance(result, str)


class TestNormalizeSpecimen:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("nasopharyngeal swab", "NP"),
            ("NP swab", "NP"),
            ("anterior nasal swab", "NASAL"),
            ("nasal swab", "NASAL"),
            ("nasopharyngeal and oropharyngeal swab", "COMBINED_NT"),
            ("saliva", "SALIVA"),
            ("", "UNKNOWN"),
            ("blood", "UNKNOWN"),  # not a respiratory specimen
        ],
    )
    def test_specimen_mapping(self, raw, expected):
        assert normalize_specimen(raw) == expected, f"Input: {raw!r}"


class TestExpandLoincLcn:
    def test_naa_probe_expands_to_naat_and_pcr(self):
        result = expand_loinc_lcn("SARS-CoV-2 RNA NAA with probe detection")
        assert "NAAT" in result
        assert "PCR" in result

    def test_immunoassay_expands_to_antigen(self):
        result = expand_loinc_lcn("rapid immunoassay")
        assert "ANTIGEN" in result

    def test_idempotent_on_already_expanded(self):
        # Calling twice should not double-expand
        once = expand_loinc_lcn("SARSCOV2 NAAT")
        twice = expand_loinc_lcn(once)
        assert once == twice


# ---------------------------------------------------------------------------
# 2. Integration test — does the retrieval pipeline return rank-1
#    correct LOINC for known hard-coded examples?
#    Loads real LOINC table + fits TF-IDF. Skipped if data files absent.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def retrieval_index():
    """Build the demo retrieval index once per test module."""
    pytest.importorskip("sklearn")
    try:
        loinc = pd.read_csv("data/processed/covid_surveillance_loinc.csv")
    except FileNotFoundError:
        pytest.skip("LOINC reference file not available")

    from src.clinical_utils import clean_text
    from src.model_building_utils import (
        expand_loinc_lcn,
        build_corpus,
        compute_relatednames_stopwords,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    loinc = loinc[~loinc["method_typ"].isna()].copy()
    loinc["expanded_lcn"] = (
        loinc["long_common_name"].map(clean_text).map(expand_loinc_lcn)
    )
    rn_sw = compute_relatednames_stopwords(loinc, threshold=0.85)

    corpus = build_corpus(loinc, "lcn_method_dict_combined", rn_sw)
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True)
    matrix = normalize(vec.fit_transform(corpus))

    return {"loinc": loinc, "matrix": matrix, "vec": vec}


def _retrieve(query: str, index: dict, top_k: int = 5) -> list[str]:
    """Return top-k predicted LOINC codes for a query string."""
    from src.clinical_utils import clean_text
    from src.model_building_utils import normalize_elr
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    q = normalize_elr(clean_text(query))
    qvec = normalize(index["vec"].transform([q]))
    scores = cosine_similarity(qvec, index["matrix"]).flatten()
    top_idx = scores.argsort()[::-1][:top_k]
    return index["loinc"]["loinc_num"].iloc[top_idx].tolist()


# Known curated examples — same three used in the Streamlit demo tab.
# These are regression tests: if the retrieval logic changes and rank-1
# drops for any of these, CI fails and you know immediately.
CURATED_EXAMPLES = [
    (
        "COVID-19 RDRP GENE FINAL - EXPECTORATED SPUTUM",  # noisy RdRp example
        "94534-5",  # SARS-CoV-2 RNA NAA NP
    ),
    (
        "LUMIRADX INFLUENZA B RT-PCR NASOPHARYNGEAL SWAB",  # vendor-brand influenza
        "76080-1",  # SARS + Flu A+B Ag combo
    ),
    (
        "SARS-COV-2 PCR NASOPHARYNGEAL ASPIRATE SWAB",  # clean NAAT example
        "94759-8",
    ),
]


@pytest.mark.parametrize("query,expected_loinc", CURATED_EXAMPLES)
def test_curated_rank1(query, expected_loinc, retrieval_index):
    """
    Regression test: best config must return correct LOINC at rank 1
    for each curated demo example.
    """
    results = _retrieve(query, retrieval_index, top_k=5)
    assert results[0] == expected_loinc, (
        f"Query: {query!r}\nExpected rank-1: {expected_loinc}\nGot: {results}"
    )


# ---------------------------------------------------------------------------
# 3. Summary parquet schema tests — run in CI before deploy
#    Separate from validate_summaries.py (which is a script);
#    these run inside pytest so failures show up in the test report.
# ---------------------------------------------------------------------------

EXPECTED_SCHEMAS = {
    "data/results/summary/primary_by_config.parquet": [
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
    "data/results/summary/filter_by_config.parquet": [
        "filter_applied",
        "corpus_strategy",
        "mrr_grouped",
        "top1",
        "top3",
        "top5",
        "n",
    ],
    "data/results/summary/filter_by_loinc.parquet": [
        "filter_applied",
        "true_loinc",
        "mrr_grouped",
        "n",
    ],
    "data/results/summary/st_by_config.parquet": [
        "model_type",
        "strategy",
        "mrr_grouped",
        "top1",
        "top3",
        "n",
    ],
    "data/results/summary/primary_by_coverage_noise.parquet": [
        "corpus_strategy",
        "model_desc",
        "n_distractors",
        "coverage_pattern",
        "noise_level",
        "mrr_grouped",
        "n",
    ],
}


@pytest.mark.parametrize("path,required_cols", EXPECTED_SCHEMAS.items())
def test_parquet_schema(path, required_cols):
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        pytest.skip(f"Summary file not yet generated: {path}")
    missing = set(required_cols) - set(df.columns)
    assert not missing, f"{path}: missing columns {missing}"
    assert len(df) > 0, f"{path}: empty dataframe"
