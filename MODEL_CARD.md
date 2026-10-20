# Model card - CORDIS field classifier

## What it does

Given the title and objective of a research project, returns the EuroSciVoc
second-level scientific fields it belongs to (37 fields, several per project),
each with a score and the decision threshold used.

## Intended use

- Triage and search: suggesting fields for new project descriptions, routing
  documents to reviewers, faceting a portfolio.
- A suggestion for a person to confirm, not an authoritative label.

## Not intended for

- Funding, eligibility or evaluation decisions about a proposal.
- Texts that are not research-project descriptions (news, patents, papers): the
  model has only seen CORDIS objectives and will still return confident scores.
- Non-English text.

## Training data

CORDIS Horizon 2020 projects (2014-2020), official bulk export from
data.europa.eu, reused under the European Commission reuse notice
(Decision 2011/833/EU). 25,801 training / 3,159 validation / 3,261 test
projects, split by a hash of the normalised text so duplicated descriptions
never straddle splits.

Labels are the EuroSciVoc paths that CORDIS attaches to each project, truncated
to level 2. CORDIS marks most of these as **not human-validated** (only 725 H2020
projects carry the human-validated flag), so the labels are themselves partly
automatic and noisy. Scores against them measure agreement with CORDIS's
labelling, not ground truth about the science.

## Models

The served default is TF-IDF + logistic regression (CPU, 0.3 ms per document,
H2020 test micro-F1 0.663). The fine-tuned `BAAI/bge-small-en-v1.5` (33M
parameters, first 256 tokens, temperature 1.05) is more accurate (0.709) and more
robust to the later programme (0.709 vs 0.637), but needs PyTorch and runs at
11 ms per document on an Apple M4 GPU.

## Evaluation

All thresholds are tuned on H2020 validation only. Three held-out sets:

| set | what it tests |
| --- | --- |
| H2020 test | in-distribution generalisation |
| Horizon Europe (2021-2027) | a later programme: new topics, new wording, a different labelling density |
| Horizon Europe, human-validated subset | the same, restricted to labels CORDIS marks as human-checked |

Headline numbers, per-field results and calibration are in the README and
`reports/`. Hyper-parameters, epochs and thresholds are selected on H2020
validation; the test and Horizon Europe scores are reported, never used for
selection.

## Known limitations

- **Rare fields are unreliable.** F1 tracks training support; for the fine-tuned
  model five fields (fewer than ~400 training projects, or catch-all "other ..."
  classes) have F1 below 0.5. See `reports/error_analysis.md`.
- **Distribution shift.** On Horizon Europe the fine-tuned model keeps its
  micro-F1 (0.709) but macro-F1 drops from 0.627 to 0.608 and macro PR-AUC from
  0.665 to 0.622; it predicts 2.47 fields per project where the newer labelling
  assigns 2.13. Re-tuning thresholds on up to 4,042 labelled Horizon projects
  changes micro-F1 by at most 0.6 points (`reports/adaptation.md`). Retraining on
  the newer programme is the remedy.
- **Field confusions follow real overlap**: computer science is the most common
  wrong addition (it co-occurs with almost every applied field), and biology /
  basic medicine / clinical medicine are frequently exchanged.
- Fine-tuned models see the first 256 tokens; long objectives are truncated.

## Ethical considerations

The data describes publicly funded projects and names no individuals in the text
fields used. A model that systematically under-labels a field (here: humanities
and media studies, which are rare in the training data) could make those
projects harder to find if used for search; keep a person in the loop.
