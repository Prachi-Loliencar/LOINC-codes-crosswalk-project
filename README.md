# LOINC Crosswalk on Simulated ELR Data

A clinical NLP retrieval system that maps noisy Electronic Lab Report (ELR)
test name strings to standardized LOINC codes. The project simulates realistic
ELR data from the CDC LIVD table, benchmarks TF-IDF and sentence-transformer
retrieval architectures under controlled noise conditions, and quantifies the
contribution of structured metadata to retrieval accuracy.

**[Live demo → Streamlit app](#)** *(link once deployed)*

---

## Results at a glance

The best TF-IDF configuration achieves **grouped MRR 0.747** on a held-out
validation set of 5,280 simulated ELR strings across 36 LOINC codes,
outperforming the best sentence transformer (S-PubMedBERT-MS-MARCO, MRR 0.617)
by 13 percentage points. Full results, ablation details, and model selection
rationale are in the [Results](#results) section.

---

## Why this problem matters

LOINC crosswalking is a persistent pain point in health data interoperability.
ELR reporting, HIE onboarding, and lab data normalization all require mapping
free-text or semi-structured test names to a controlled vocabulary. This project addresses it as an information retrieval problem,
using the CDC LIVD device submission table as a principled source of real-world
naming variation.

---

## Data sources

**CDC LIVD Table.** Laboratory In Vitro Diagnostics device submissions for
SARS-CoV-2, containing FDA-authorized test kit mappings to LOINC codes via
vendor analyte names, specimen descriptions, and method information. The LIVD
table determines scope: any LOINC code referenced in its submissions is
included, encompassing single-analyte COVID codes and combination respiratory
panels (flu A/B, RSV, SARS-CoV-2). 

**LOINC Table.** Full LOINC table joined to LIVD on LOINC code, providing
component, system, method_typ, scale_typ, and long_common_name for corpus
construction.

Raw data files are not included in this repository. The LIVD table is available
from the [CDC SARS-CoV-2 LIVD page](https://www.cdc.gov/csels/dls/livd-codes.html)
and the LOINC table from [loinc.org](https://loinc.org/downloads/) (free
registration required).

---


## Simulation design
 
**Deduplication.** The raw LIVD table contains approximately 998 device
submission rows after merging with the LOINC table. Many LIVD rows list
multiple valid specimen types for a single device (e.g. "Nasopharyngeal swab
or Nasal swab"), stored as newline-separated entries in the Vendor Specimen
Description field. These are exploded into one row per specimen type before
deduplication, producing 1,829 rows across 36 LOINC codes. This explosion is
necessary because specimen type is a simulation axis, each specimen type
produces distinct ELR strings, and collapsing them would artificially limit
naming variation.
 
The 1,829 rows are then deduplicated on a clinical key combining component,
method_typ, system, and vendor analyte name. This removes manufacturer-driven
redundancy (multiple devices submitting identical vendor analyte names for the
same test) while preserving rows where different vendors use different
names - the primary source of surface form variation in the simulation.
Post-deduplication: 642 rows. After a minimum-seeds filter (≥3 LIVD rows per
LOINC code to ensure sufficient perturbation diversity): 556 seeds across 36
codes.
 
**Noise taxonomy.** Three noise dimensions are tracked independently per
simulated string. Categories are defined on input transformations independently
of any retrieval model or corpus and describe what happened to the string,
not the retrieval consequence:
 
- *Corruption*: character-level damage (typos: swap, skip, extra character).
  Token identity is partially or fully destroyed. Implemented via decaying
  probability typo injection; space characters are excluded since space
  insertion is not a realistic keystroke error in structured instrument fields.
- *Compression*: a signal token is replaced with an alternate surface form of
  the same semantic entity. The information is still present but encoded
  differently, and is recoverable by a domain-aware model. Examples: `SARS-CoV-2` → `COVID-19` or `CORONAVIRUS` (same pathogen, different surface form); `RNA` → `NAA` or `PCR` (same detection chemistry); `nucleocapsid` → `N-GENE` (same gene target).
- *Omission*: signal deleted entirely. Either a token is replaced with an
  empty string, or an entire component (method, specimen) is structurally
  absent from the template. Unrecoverable without external metadata or a
  retrieval-side expansion dictionary.
Interpretation tokens (`STATUS`, `RESULT`, `FINAL`) are appended at 10%
probability to simulate LIS field verbosity but are not counted toward any
noise dimension since they do not damage, substitute, or remove signal tokens,
and their retrieval impact is negligible (confirmed by zero weight on
`has_interp` in the coverage pattern scoring and near-zero IDF in the corpus).
 
**Template configuration.** ELR strings are assembled from up to four
components, namely: model name, analyte, method, and specimen, under six weighted
templates. Weights represent an assumed prior over ELR field completeness
(no ground truth distribution was available). The dominant templates are
analyte+method+specimen (30%) and analyte+method (30%), reflecting the
frequency with which older LIS systems fail to populate the specimen
segment.
 
**Coverage patterns.** Each simulated string is scored for which signal types
are present: A (analyte), M (method), S (specimen), I (interpretation token).
The most common patterns are `A+M` (41%), `A+M+S` (22%), and `A` alone (12%).
 
---
 
## Retrieval architecture
 
```
CDC LIVD Table ──► Preprocessing ──► Clean Seeds (551 rows, 36 LOINC codes)
                                              │
                                       ELR Simulation
                                   (12 variants × noise injection)
                                              │
                                      6,600 ELR strings
                                              │
                           ┌──────────────────┴──────────────────┐
                      TF-IDF retrieval              Sentence transformer
                  (corpus strategy ablation)    (6 models × 2 corpus strategies)
                           │
                     Post-retrieval filters
                     (oracle / brand imputation)
                           │
                   Specimen-aware grouped MRR evaluation
```
 
**Stage 1 — Simulation.** LIVD vendor analyte names are preprocessed,
deduplicated on a clinical key, and expanded to 12 stochastic variants per
seed via the three-axis noise taxonomy described above.
 
**Stage 2 — Corpus construction.** The 98-code LOINC reference panel is
enriched with domain expansion dictionaries (`LOINC_SYSTEM_EXPANSION`,
`LOINC_METHOD_EXPANSION`, `LOINC_LCN_EXPANSION`) that map formal LOINC
terminology to the surface forms observed in real ELR strings. Several corpus
strategies are ablated; the best-performing strategy (`lcn_method_dict_combined`)
concatenates the expanded long common name with system and method dictionary
expansions.
 
**Stage 3 — Retrieval.** ELR queries are normalized (eg.`clean_text` +
`normalize_elr` for TF-IDF) and matched against the corpus via cosine similarity over
TF-IDF or sentence-transformer embeddings. Three vectorizer types are evaluated:
word n-grams, character n-grams, and a mixed model that combines both via a
weighted sum of their L2-normalized matrices. In the mixed model, α controls
the relative contribution of the word component (α) versus the character
component (1−α), where both sub-matrices are L2-normalized before scaling so
α maps linearly to cosine-space contribution.
 
### Corpus strategies
 
The primary ablation compares five corpus construction strategies. The corpus
is always the same 36 LOINC codes; only the text representation varies:
 
- **`lcn_only`**: Long common name (LCN) only (baseline). Example: `"SARS-CoV-2
  (COVID-19) RNA [Presence] in Respiratory specimen by NAA with probe
  detection"`.
- **`combined`**:  Long common name (repeated 2x) + system expansion +
  relatednames2 (synonym field). The LCN is repeated to counteract signal dilution from the
  relatednames2 field, which is long and heterogeneous - its tokens inflate
  raw term frequency counts and reduce the relative weight of discriminative
  LCN tokens. String repetition raises TF counts for LCN tokens without
  affecting IDF, providing a partial counterbalance.
- **`lcn_method_dict_combined`** *(overall best)*: Long common name + system
  expansion + system expansion + method dictionary expansion, without LCN repetition.. Replaces generic LOINC system and
  method values with domain-specific surface forms. For example, `Nph` expands
  to `"Nasopharynx Nasopharyngeal NP NPH"`, and `Probe.amp.tar` expands to
  `"NAAT NAA PCR RT-PCR QPCR"`. This explicitly bridges the gap between formal
  LOINC terminology and the vernacular names used in ELR strings. The fact that this strategy outperforms `combined`
  without needing LCN repetition indicates that replacing relatednames2 with
  a compact method expansion dictionary eliminates the dilution problem at its
  source rather than compensating for it indirectly.
- **`lcn_method_dict_filtered_rn`**: A combination of the last two; Long common name (repeated 2x) + system
  expansion + method dictionary + filtered relatednames2. Filters relatednames2
  to remove tokens appearing in >85% of codes (treated as uninformative
  stopwords) before including it alongside the method dictionary. Retains some
  relatednames2 signal while reducing dilution, but performs below
  `lcn_method_dict_combined`, suggesting the residual relatednames2 content
  adds noise even after filtering.
- **`component_weighted_method_dict`**: Component field (repeated 2x via
  string concatenation) + method dictionary expansion + system expansion.
  Upweights the LOINC component axis on the assumption that it is more
  discriminative than the full long common name. Performs well with distractors
  but shows unstable generalization behavior as distractor count increases.

  
Additional strategies were explored during development
but are not included in the primary ablation. These variants tested lower and higher LCN/component repetition counts (eg. 3x) and combinations thereof; all performed below
their lower-repetition counterparts (with the exception of `component_weighted_method_dict` which performed worse with component x1), confirming that string repetition is an
indirect and unreliable substitute for explicit vocabulary expansion. These
strategies have been removed from `src/model_building_utils.py` to keep the
codebase aligned with the repo.

### Post-retrieval filters
 
Two optional post-retrieval reranking strategies are evaluated to quantify
the value of structured metadata:
 
**Oracle filter (upper bound).** Uses ground-truth method class (NAAT vs antigen) and specimen
type from the simulated ELR row to demote mismatching candidates with a 0.5x
penalty, then re-ranks. This represents the maximum possible gain from
metadata filtering if field extraction were perfect — an unrealistic but
useful ceiling.
 
**Brand filter (production-feasible).** Scans the ELR string for known
instrument brand tokens (e.g. `COBAS`, `VERITOR`, `XPERT`) and imputes method
class via a hand-curated lookup table (e.g. `COBAS` → `probe.amp.tar`). Applies
the same demotion logic as oracle filter. This is feasible in production
because it requires only the ELR string itself.
 
---
 
## Evaluation
 
**Primary metric.** Specimen-aware grouped MRR (`mrr_grouped`). For each ELR
string, the valid answer set is expanded beyond the single true LOINC code to
include LOINC codes sharing the same component and method with a
specimen-compatible system value, and gene-target ambiguous codes where vendor
analyte names provide insufficient token coverage to distinguish between codes.
Reciprocal rank is computed against this expanded valid set. This correctly
handles the LOINC design choice of using generic respiratory specimen codes
for COVID-19 reporting rather than specimen-specific ones, for example, 45.5% of wrong
top-1 predictions are specimen specificity mismatches (such as predicting Nose vs Respiratory System) absorbed by the grouped metric rather than genuine retrieval failures.
 
**Ablation structure.** Four nested stages:
 
1. *Primary* — corpus strategy × model type × distractor count
2. *Secondary* — finer sweep over mixed word+char n-gram ranges and alpha weights
3. *Filter ablation* — no-filter vs oracle vs brand-imputation post-retrieval
   reranking on the best configuration
4. *Sentence transformer* — 6 models × 2 corpus strategies, with a separate
   preprocessing pipeline that preserves natural language structure rather than
   applying TF-IDF tokenization
**Splits.** Train/val/test is assigned by variant number rather than randomly
within LOINC code, reflecting the deployment scenario where all LOINC codes
are known at inference time and generalization is over novel surface forms from
unseen lab senders.
 
---
 
## Results
 
All evaluation uses **specimen-aware grouped MRR** on a held-out validation
set of 5,280 simulated ELR strings across 36 LOINC codes.
 
| Configuration | Grouped MRR |
|---|---|
| TF-IDF — `lcn_method_dict_combined`, word unigram, 0 distractors | **0.747** |
| TF-IDF — oracle filter upper bound (perfect metadata) | 0.767 |
| TF-IDF — brand filter (production-feasible) | 0.748 |
| Best sentence transformer (S-PubMedBERT-MS-MARCO) | 0.617 |
 
TF-IDF leads the best sentence transformer by **+13 percentage points** overall,
widening to **+21 pp** on analyte-only strings (coverage pattern `A`) where
dense encoders fail to distinguish method signal absent from the query. The
oracle filter ceiling of 0.767 confirms that the remaining gap is not
recoverable from metadata filtering alone — it reflects genuine retrieval
ambiguity in the corpus. The negligible gain from the brand filter (0.747 →
0.748) reflects the corpus composition: 59 of 98 COVID LOINC codes are probe
amplification assays, so method imputation adds little discriminative power
against an already method-skewed corpus.
 
### Model selection rationale
 
The best-performing TF-IDF configuration overall (`component_weighted_method_dict`
corpus, mixed word+char model, α=0.3, 143 distractors) achieved grouped MRR
**0.760**. However, for the primary production-facing configuration, 
`lcn_method_dict_combined` was selected with a simpler word unigram model and 0 distractors
(grouped MRR **0.747**) for three reasons:
 
1. **Parsimony.** The 0.013 MRR gap (1.7 pp) does not justify the added
   complexity of a mixed vectorizer and external distractor sampling.
2. **Robustness to corpus changes.** Zero distractors means the corpus is fixed
   at deployment time. The 143-distractor config is sensitive to which
   non-COVID respiratory codes are included, making it harder to reason about
   in production.
3. **Distractor effect is unstable.** `component_weighted_method_dict` gains
   with distractors while `lcn_method_dict_combined` degrades monotonically.
   This inconsistency suggests the distractor benefit may not generalize beyond
   this specific corpus.
All subsequent analyses (filter ablation, error analysis, noise robustness)
use the simpler `lcn_method_dict_combined`, word unigram configuration.
 
### Noise robustness
 
| Noise level | TF-IDF | Best ST |
|---|---|---|
| Low  | 0.762 | 0.634 |
| Medium | 0.710 | 0.570 |
| High | 0.510 | 0.457 |
 
TF-IDF is substantially more robust, losing 25 pp from low to high noise versus
18 pp for the best sentence transformer. The low vocabulary coverage rate
between the ELR query space and the LOINC corpus (7.8%) favors sparse lexical
matching over dense semantic similarity — compression noise (surface variants
of the same entity) is largely recovered by the retrieval-side expansion
dictionaries, while omission noise (signal absent entirely) degrades both
models. Detailed per-dimension noise analysis is in
`notebooks/04_error_analysis.ipynb`.
 
Note that omission count correlates with method token absence (Pearson r =
−0.73) because structural template omission and target deletion both increment
the omission counter and both reduce method signal. Omission-stratified results
should be interpreted in the context of coverage patterns rather than as a
pure noise effect.
 
---
 
## Repository structure
 
```
loinc-crosswalk/
├── src/
│   ├── clinical_utils.py               # Domain constants, text cleaning, specimen normalization
│   ├── model_building_utils.py         # Corpus construction, TF-IDF index, retrieval, evaluation
│   ├── ablation.py                     # Ablation runners: primary, secondary, filter
│   ├── elr_simulation.py               # ELR simulation pipeline
│   ├── sentence_transformer_ablation.py  # ST model benchmarking
│   ├── error_analysis.py               # Error classification and visualization
│   ├── corpus_and_simulation_viz.py    # UMAP, similarity, and simulation visualizations
│   ├── livd_and_loinc_preprocessing.py # Raw data loading, merging, and filtering
│   └── __init__.py
├── notebooks/
│   ├── ablation_results_combined.ipynb
│   ├── corpus_simulation_viz.ipynb
│   ├── error_analysis.ipynb
│   └── test_set_evaluation.ipynb
├── app/
│   └── app.py                          # Streamlit portfolio dashboard
├── logs/                               # Runtime logs (gitignored)
├── data/                               # Raw and processed data (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```
 
---
 
## Setup and reproduction
 
```bash
git clone https://github.com/<your-username>/loinc-crosswalk.git
cd loinc-crosswalk
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
Place the LIVD and LOINC source files in `data/raw/` per the paths expected
in `src/livd_and_loinc_preprocessing.py`, then run the pipeline in order:
 
```bash
python src/livd_and_loinc_preprocessing.py  # produces data/processed/
python src/elr_simulation.py                # produces elr_simulated.csv
python src/ablation.py                      # produces primary and filter ablation CSVs
python src/sentence_transformer_ablation.py # produces ST ablation CSV
```
 
Analysis notebooks in `notebooks/` can then be run in any order. To launch
the Streamlit app locally:
 
```bash
streamlit run app/app.py
```
 
---
 
## Limitations
 
**Simulation-based evaluation.** Generalization is over perturbation variation
of known LOINC codes, not over truly novel lab submissions from unseen senders.
Performance on genuinely new analyte naming conventions may differ. Validation
against real de-identified ELR data is planned (see Future work).
 
**LIVD-bounded scope.** 36 LOINC codes covering COVID-19 and combination
respiratory panels. Non-COVID respiratory codes (influenza-only, RSV-only,
strep) are out of scope. Extension to the full respiratory panel LOINC space
is the natural next step.
 
**Specimen filtering underperforms.** The dominant LOINC system value in this
dataset is the generic `Respiratory System Specimen` catch-all, which must be
retained for any respiratory specimen type by design. Method signal carries
substantially more discriminative weight than specimen signal, this is a
domain-informed finding rather than a system limitation.
 
**Template weights are assumed priors.** No ground truth distribution of ELR
field completeness was available. CDC NNDSS or state health department ELR
intake logs could provide an empirical prior for future work.
 
---
 
## Future work
 
- Validate against real de-identified ELR submissions (MIMIC-IV access in
  progress; CITI certification completed)
- Extend to non-COVID respiratory LOINC codes
- Fine-tune a sentence transformer on domain-specific clinical text
- Add a cloud analytics layer (BigQuery + dbt) as a companion forecasting
  project
---
 
## Acknowledgements
 
LOINC codes and terminology are provided by the Regenstrief Institute and
are used in accordance with the [LOINC license](https://loinc.org/license/).
The CDC LIVD table for SARS-CoV-2 is a public domain resource made available
by the Centers for Disease Control and Prevention. This project is not
affiliated with or endorsed by either organization.
 
---

 