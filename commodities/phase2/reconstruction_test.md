# Reproducibility test of the narrative features

Re-implementation of the team's construction from saved inputs (see `reconstruct_features.py`). Missing exposures are filled by random donor rows in the team's pipeline, so only rows with complete exposures can be compared exactly.

## Part 1. 5-day window, against the saved feature block

| model | topics | rows compared | share of all rows | mean per-date Spearman | minimum per-date Spearman | median absolute difference | share within 0.02 |
|---|---|---|---|---|---|---|---|
| nolbert20 | 20 | 101,452 | 99.0% | 1.0000 | 1.0000 | 0.00101 | 100.0% |
| finbert | 20 | 101,452 | 99.0% | 1.0000 | 1.0000 | 0.00101 | 100.0% |
| lda_all | 60 | 101,452 | 99.0% | 1.0000 | 1.0000 | 0.00101 | 100.0% |

## Part 2. Saved factor returns reproduced from reconstructed features and saved loadings

Factor return of month t = least-squares coefficient of next-period relative returns on (features x loadings). Comparison with the saved `factors.csv`.

| specification | window | K | months | correlation with saved factors (min over factors) | slope (range) | test months: min correlation |
|---|---|---|---|---|---|---|
| `narr_nolbert20__sipca` | 5 | 1 | 104 | 0.9996 | 0.94 to 0.94 | 0.9997 |
| `narr_nolbert20_s63__sipca` | 63 | 3 | 104 | 0.9989 | 0.97 to 0.98 | 0.9962 |
| `narr_finbert__sipca` | 5 | 3 | 104 | 0.9992 | 0.95 to 0.96 | 0.9987 |
| `narr_finbert_s63__sipca` | 63 | 3 | 104 | 0.9986 | 0.97 to 0.98 | 0.9975 |
| `narr_lda_all__sipca` | 5 | 1 | 104 | 0.9996 | 0.95 to 0.95 | 0.9996 |
| `narr_lda_all_s63__sipca` | 63 | 3 | 104 | 0.9979 | 0.89 to 0.91 | 0.9934 |

Reading: part 1 shows whether the re-implementation reproduces features that exist in the saved data; part 2 shows how well it reproduces the saved factor returns, for the 5-day models (where part 1 applies) and for the 63-day models (where it is the only check available).