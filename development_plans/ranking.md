# Ranking trees via the Plackett-Luce score at the null

## Objectives

1. Add a `RankingTree` estimator to Sigma that fits ctree-style conditional
   inference trees on responses that are full or partial rankings of K items.
2. Reuse the existing influence-function abstraction in `sigma/_tree.py`
   (lines 850-978) without introducing MOB-style per-node refitting.
3. Ground the splitting criterion in a SOTA learning-to-rank loss. The chosen
   per-observation transformation h(Y_i) must be mathematically equivalent (or
   first-order equivalent) to that loss at the null hypothesis of equal worth.
4. Keep the design parallel to `SurvivalTree` / `compute_logrank_scores`: a pure
   per-sample influence function, a leaf-level summary statistic, and a CI
   helper, all in a new `sigma/_ranking.py` mirroring `sigma/_survival.py`.

## Analysis

### What Hothorn published on ranking trees

Not a ctree treatment. The two ranking-tree methods in his ecosystem use **MOB**
(model-based recursive partitioning), not ctree:

- Bradley-Terry trees: Strobl, Wickelmaier, Zeileis (2011), `bttree` in
  `psychotree`. Paired comparisons.
- Plackett-Luce trees: Turner et al., `pltree` in `PlackettLuce`. Full rankings.

The 2006 ctree paper (Hothorn-Hornik-Zeileis) and the partykit `ctree.Rnw`
vignette enumerate supported response types as univariate continuous, censored,
J-class classification, ordinal, multivariate. Ranking-data is absent. The
"ranking" influence function in the vignette is the Wilcoxon rank transform of
a *scalar* response, not a transformation of a ranking-data response.

The structural reason: ctree needs a fixed per-observation h: Y -> R^q. For
ranking responses, the natural per-observation contribution is the gradient of
a fitted PL/BT log-likelihood at node-specific MLE parameters, which is exactly
what MOB does and what ctree cannot accommodate as a static transformation.

### SOTA loss survey

| Loss                              | Type                       | Per-obs score at null                  | Suitability |
| --------------------------------- | -------------------------- | -------------------------------------- | ----------- |
| ListMLE (Xia et al., ICML 2008)   | listwise, PL likelihood    | clean, K-dim                           | chosen      |
| ListNet (Cao et al., ICML 2007)   | listwise top-1 PL          | K-dim, equivalent to ListMLE for top-1 | special case|
| LambdaRank/LambdaMART             | NDCG-weighted pairwise     | NDCG-bound, no clean log-lik gradient  | rejected    |
| LambdaLoss (Wang et al., 2018)    | proper LR on NDCG          | requires graded relevance              | rejected    |
| ApproxNDCG                        | smooth NDCG                | requires graded relevance              | rejected    |
| RankNet (Burges 2005)             | pairwise BT                | K(K-1)/2 dim                           | scales poorly |
| Position-Aware ListMLE            | weighted PL                | weighted PL score                      | future extension |

ListMLE wins because it is the negative log-likelihood of the Plackett-Luce
model (the canonical generative model for full and partial rankings), it admits
a closed-form per-observation gradient at the uniform null, and the metric-bound
losses (LambdaRank/LambdaLoss/ApproxNDCG) require graded-relevance labels that
are out of scope for ranking-data responses.

### The chosen transformation: Plackett-Luce score at the uniform null

For a ranking i_1 succ i_2 succ ... succ i_J (J <= K, partial allowed) with
worth parameters alpha_k = exp(theta_k), the PL log-likelihood is:

```
log P(pi_i | theta) = sum_{j=1}^J [ theta_{i_j} - log( sum_{l in A_j} exp(theta_l) ) ]
```

where A_j is the at-risk set at stage j (items not yet placed). Source:
PlackettLuce package vignette,
https://cran.rstudio.com/web/packages/PlackettLuce/vignettes/Overview.html.

Score wrt theta_k:

```
d log P / d theta_k = sum_{j=1}^J [ I(i_j = k) - I(k in A_j) * exp(theta_k) / sum_{l in A_j} exp(theta_l) ]
```

At theta = 0 the inner sum is |A_j| = K - j + 1. With harmonic numbers
H_n = sum_{m=1}^n 1/m and H_0 = 0, for a complete ranking:

```
S_{ik}(0) = 1 - ( H_K - H_{K - r_i(k)} )
```

where r_i(k) is the position of item k in ranking i (1 = first). For top-d
partial rankings, items at positions 1..d use the same formula; items that did
not appear get:

```
S_{ik}(0) = -( H_K - H_{K - d} )
```

Sanity checks:

- K=2 winner: S = 1 - 1/2 = +1/2; loser: S = 1 - 3/2 = -1/2. Sum 0.
- K=3 ranking (a, b, c): scores (2/3, 1/6, -5/6). Sum 0.
- Per-ranking sum is always 0 (PL identifiability constraint).

### Why this transformation is the right choice

1. **First-order equivalent to ListMLE.** ListMLE loss is -log P(pi_i | theta);
   S_i(0) is exactly -grad ListMLE evaluated at theta=0. Splitting on
   independence between S_i(0) and a covariate is the **Rao score test** for
   H_0: covariate has no effect on PL worths, asymptotically equivalent to the
   LRT used by `pltree` (MOB), but cheap because no in-node MLE is needed.
2. **Exact analog of the logrank score.** The logrank score
   u_i = delta_i - sum_{t_j <= t_i} 1/n_j is the same recipe ("indicator minus
   cumulative reciprocal of at-risk count") instantiated on the implicit
   ranking of subjects by death time. PL-score-at-0 generalizes logrank from a
   1D ranking-by-time to a K-D ranking over items. This is the same
   Cox-partial-likelihood / ctree-framework parallel Hothorn et al. (2006) drew
   for survival.
3. **Compact and fixed.** h(Y_i) in R^K is computable once at fit-time, does
   not need to be recomputed per node, fits Sigma's existing static-influence
   pattern.
4. **Handles partial rankings, top-d, and ties** naturally (top-d by truncating
   the j-sum; ties via Davidson's extension or rank-averaging).

### What the transformation is not

- Not a substitute for fitting a PL model at leaves. It is the *splitting*
  criterion. Leaf prediction defaults to the empirical weighted mean rank
  vector per item (cheap, interpretable). A PL-MLE leaf summary can be added
  later via Sigma's transmuter pattern (`_tree.py:991-1038`).
- Not identical to `pltree`. `pltree` (MOB) refits PL per node; we use the
  null score and never refit. They agree asymptotically under H_0 and diverge
  under strong alternatives. The trade is simplicity and speed for asymptotic
  fidelity.

### Critical files in the existing codebase

| Path                          | Role                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `sigma/_tree.py:850-978`      | Abstract `Tree` methods to override                           |
| `sigma/_tree.py:2722-3200`    | `SurvivalTree` template to mirror                             |
| `sigma/_tree.py:991-1038`     | Transmuter pattern (for optional PL-MLE leaf summary later)   |
| `sigma/_survival.py:16-55`    | `compute_logrank_scores` template                             |
| `sigma/_node.py:196-300`      | `SurvivalNode` metric-rendering pattern                       |
| `sigma/__init__.py`           | Public exports                                                |

## Numbered actions

1. **Write the influence-function tests first.** Add `tests/test_ranking.py`
   with unit tests covering, in this order:
   1. K=2 hand-computed scores (winner +1/2, loser -1/2).
   2. K=3 hand-computed scores ((2/3, 1/6, -5/6) for ranking (a,b,c)).
   3. Per-ranking sum-to-zero invariant for random K up to 10.
   4. Top-d partial-ranking formula on hand-computed examples.
   5. **Numerical equivalence to `compute_logrank_scores`** when the input is
      a death-order ranking (no censoring, no ties): assert element-wise
      equality between `compute_pl_null_scores` on the implicit subject
      permutation and `compute_logrank_scores` on the same data. This proves
      the mathematical correspondence claimed in the analysis.
   These tests must fail with `ImportError` until step 2 lands.
2. **Implement `sigma/_ranking.py`** with:
   1. `compute_pl_null_scores(rankings, n_items, weights=None) -> NDArray[float]`
      returning shape `(n_obs, n_items)`. Input convention: `rankings[i]` is an
      int array of length d_i <= n_items listing item indices best-to-worst,
      with `-1` (or padding) for unranked positions. API style mirrors
      `compute_logrank_scores`.
   2. `compute_mean_rank_vector(rankings, weights, n_items)` for leaf
      prediction.
   3. (Optional, deferred) `compute_pl_mle(rankings, weights, n_items)` for the
      PL-fitted transmuter path. Not required for v1.
   Run step 1 tests; they must pass.
3. **Add `RankingNode`** to `sigma/_node.py` mirroring `SurvivalNode`. Carry
   per-item `RankingMetric(label, value, ci_low, ci_high)` entries (one per
   item) for the textual and graphviz renderers.
4. **Add `RankingTree(Tree[_node.RankingNode])`** to `sigma/_tree.py`
   implementing the eight abstract methods of step (4) of the analysis:
   1. `_validate_fit_params(X, y)` enforces `y.shape == (n_obs, max_d)` with
      int dtype, values in `[-1, n_items)`, no duplicates within a row, at
      least 2 ranked positions per row.
   2. `_validate_offset(offset, n_samples)` raises (no offset support in v1).
   3. `_compute_influence(y, offset)` calls `compute_pl_null_scores`.
   4. `_compute_prediction(y, weights)` calls `compute_mean_rank_vector`.
   5. `_is_constant_response(y, weights)` returns True iff all rankings are
      element-wise identical (or PL scores are within numerical tolerance of
      zero).
   6. `_compute_ci(y, weights)` per-item bootstrap CI on mean rank.
   7. `_compute_per_class_ci`, `_compute_class_distribution`,
      `_compute_survival_function`, `_compute_survival_metrics` return `None`.
5. **Wire exports.** Add `RankingTree` and `RankingNode` to
   `sigma/__init__.py`.
6. **End-to-end smoke test.** Add a test in `tests/test_ranking.py`:
   simulate K=4 items, n=400 observations, with covariate X causally flipping
   the worth (theta = +/- delta * X). `RankingTree.fit` must split on X with
   high significance (p < 1e-3) and the leaf mean-rank vectors must reflect
   the flipped preference order on the two sides.
7. **Update `README.md`** to add ranking to the supported response types and
   include a minimal usage snippet (analogous to the existing survival
   snippet).
8. **Mark this file's actions DONE** as each step lands.
9. **Final verification.** Run `mamba run -n standard ./assemble.sh` (format,
   lint, type check, full test suite). Resolve any issue surfaced by it before
   declaring the work done. No `# type: ignore`, no `typing.Any`, no
   commented-out tests.

## Sources

- Xia, Liu, Wang, Zhang, Li (2008), ListMLE - https://icml.cc/Conferences/2008/papers/167.pdf
- PlackettLuce vignette (likelihood and gradient) - https://cran.rstudio.com/web/packages/PlackettLuce/vignettes/Overview.html
- PL for learning to rank - https://arxiv.org/abs/1909.06722
- Hothorn-Hornik-Zeileis 2006 (ctree framework) - https://www.zeileis.org/papers/Hothorn+Hornik+Zeileis-2006.pdf
- Cox-partial-likelihood / logrank score derivation - https://web.stanford.edu/~lutian/coursepdf/unitcox1.pdf
- 2024-2025 listwise loss landscape review - https://www.emergentmind.com/topics/listwise-ranking-losses
- Position-Aware ListMLE (future extension) - https://auai.org/uai2014/proceedings/individuals/164.pdf
- Strobl-Wickelmaier-Zeileis 2011 (BT trees, MOB) - https://www.zeileis.org/papers/Strobl+Wickelmaier+Zeileis-2011.pdf
- BTL hypothesis testing background - https://arxiv.org/abs/2410.08360
