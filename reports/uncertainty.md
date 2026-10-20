# Uncertainty

95% percentile intervals from 1,000 bootstrap resamples of projects.
Within a split every model is resampled with the same project draws, so
differences between models are paired.

| model | split | micro-F1 | macro-F1 |
| --- | --- | --- | --- |
| finetune:bge-small-en-v1.5 | test | 0.709 [0.700, 0.717] | 0.627 [0.611, 0.641] |
| finetune:bge-small-en-v1.5 | shift | 0.709 [0.705, 0.712] | 0.608 [0.601, 0.615] |
| tfidf_logreg | test | 0.663 [0.655, 0.670] | 0.563 [0.548, 0.577] |
| tfidf_logreg | shift | 0.637 [0.634, 0.641] | 0.532 [0.524, 0.539] |
| embed_logreg:bge-small-en-v1.5 | test | 0.595 [0.587, 0.603] | 0.495 [0.482, 0.507] |
| embed_logreg:bge-small-en-v1.5 | shift | 0.555 [0.552, 0.559] | 0.437 [0.431, 0.442] |

## Paired differences (row model minus column model)

| comparison | split | micro-F1 difference | macro-F1 difference |
| --- | --- | --- | --- |
| finetune:bge-small-en-v1.5 - tfidf_logreg | test | 0.046 [0.039, 0.053] | 0.065 [0.048, 0.080] |
| finetune:bge-small-en-v1.5 - tfidf_logreg | shift | 0.071 [0.068, 0.075] | 0.076 [0.069, 0.083] |
| finetune:bge-small-en-v1.5 - embed_logreg:bge-small-en-v1.5 | test | 0.113 [0.105, 0.122] | 0.132 [0.115, 0.148] |
| finetune:bge-small-en-v1.5 - embed_logreg:bge-small-en-v1.5 | shift | 0.153 [0.150, 0.157] | 0.172 [0.164, 0.179] |

## Shift drop (Horizon Europe minus H2020 test, independent samples)

| model | micro-F1 change | macro-F1 change |
| --- | --- | --- |
| finetune:bge-small-en-v1.5 | 0.000 [-0.008, 0.009] | -0.019 [-0.036, -0.001] |
| tfidf_logreg | -0.025 [-0.034, -0.017] | -0.031 [-0.047, -0.014] |
| embed_logreg:bge-small-en-v1.5 | -0.040 [-0.049, -0.031] | -0.058 [-0.072, -0.044] |
