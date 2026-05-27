# SOTA review: Sigma vs the eleven papers cited on its product page

## Objective

The Sigma product page lists eleven academic references under "Grounded in
Research", each with a one-sentence synthesis tying the paper to a specific
Sigma feature. This document audits, **paper by paper and recommendation by
recommendation**, whether Sigma's actual implementation honours those
papers' explicit advice. For every recommendation found in a paper, the
implementation is scored 🟡 YELLOW or 🔴 RED (this filtered view hides all
🟢 GREEN rows). The document closes with a numbered action plan that
resolves every YELLOW and RED finding.

Remove each resolved points so that we keep here ONLY the deficiencies that still exist.

## Scoring scale

| Symbol | Meaning |
| ------ | ------- |
| 🟡 YELLOW | Sigma is silent: the recommendation is neither honored nor contradicted - typically a missing feature, missing default, or missing documentation. |
| 🔴 RED    | Sigma uses an alternative the paper specifically argues against, or picks the wrong default among options Sigma itself exposes. |

## Per-paper review

### 5. Hothorn, Hornik, van de Wiel, & Zeileis (2006) - Lego system

**Citation.** Hothorn, T., Hornik, K., van de Wiel, M. A., & Zeileis, A.
(2006). *A Lego System for Conditional Inference.* *The American
Statistician*, 60(3), 257-263. doi:10.1198/000313006X118430

**Website synthesis.**
> "The modular permutation-test design - test statistic, null
> distribution, multiplicity correction - that Sigma's split-selection
> engine implements."

**Key recommendations from the paper.**

1. Decompose every conditional inference test into four interchangeable
   bricks: an *influence function* h(Y), a *transformation* g(X), a
   *test statistic functional* (e.g. maximum or quadratic form on T), and
   a *null distribution* (asymptotic or permutation).
2. Allow the user to plug different influence and transformation
   functions to obtain different test families (Wilcoxon, log-rank,
   chi-squared, etc.) from the same engine.
3. Use the conditional expectation `mu` and conditional covariance
   `Sigma` of T under the null to construct standardized statistics.

**Implementation audit.**

| #   | Recommendation                                                       | Status | Sigma evidence |
| --- | -------------------------------------------------------------------- | ------ | -------------- |
| 5.2 | Allow user-pluggable influence and transformation functions          | 🟡 YELLOW | The user-facing API exposes only `correlation` (`normal` / `rank`) at `sigma/_tree.py:1334`. The internal h/g are hard-coded per task; no public hook accepts arbitrary callables. |
| 5.4 | Cite the Lego framework where the modular design is implemented      | 🟡 YELLOW | The 2006 ctree paper is cited in `sigma/_statistics.py:1-7`; the Lego paper is not referenced anywhere in the source tree. |

**Verdict.** Sigma's internal architecture follows the Lego philosophy
but neither exposes the bricks for user extension nor cites the paper
that named the design.

---

### 7. Hothorn & Lausen (2003) - maximally selected rank statistics

**Citation.** Hothorn, T., & Lausen, B. (2003). *On the Exact Distribution
of Maximally Selected Rank Statistics.* *Computational Statistics & Data
Analysis*, 43(2), 121-137. doi:10.1016/S0167-9473(02)00225-6

**Website synthesis.**
> "The exact and Monte-Carlo distributions of maximally selected rank
> statistics underpin Sigma's min-P resampling option for multiplicity
> correction across candidate variables."

**Key recommendations from the paper.**

1. Use the *exact* distribution of the maximally-selected rank statistic
   when feasible (small to moderate n), via the recursive linear-rank
   algorithm extended to the sup.
2. Fall back to a Monte-Carlo permutation approximation otherwise.
3. Prefer either of the above to the improved-Bonferroni and asymptotic
   Gaussian-process bounds historically used.
4. Treat cutpoint selection as the supremum over candidate cutpoints of
   the same linear test statistic family, not as a separate procedure.

**Implementation audit.**

| #   | Recommendation                                                     | Status | Sigma evidence |
| --- | ------------------------------------------------------------------ | ------ | -------------- |
| 7.1 | Exact distribution of the maximum statistic                        | 🟡 YELLOW | Not implemented; Sigma relies on Monte-Carlo for non-asymptotic correction. |
| 7.4 | Cite Hothorn & Lausen (2003) where the supremum statistic is used  | 🟡 YELLOW | The 2006 ctree paper is cited in `_statistics.py:1-7`; the 2003 paper is not. |

**Verdict.** The Monte-Carlo path matches the paper's fallback
recommendation, but the exact-distribution path the paper centers on is
not present, and the source does not name the paper.

---

## Numbered action plan

Each entry resolves one or more YELLOW / RED rows from the audit above.
Items are ordered by user-visible impact, lowest-effort first.

### 7. Accept user-supplied influence functions

- **Target file(s).** `sigma/_statistics.py:18-188`, `sigma/_tree.py:114-148`
  (`Tree.__init__`).
- **Change.** Add an `influence: None | typing.Callable = None` parameter
  to `Tree.__init__`. When provided, use it to compute `h(Y)` instead of
  the hard-coded identity influence.
- **Success criterion.** A unit test plugs in a Wilcoxon rank influence
  and recovers a known partykit result for a non-identity test family.
- **Paper(s) addressed.** 5.2 (YELLOW).

### 8. Implement the exact distribution of the maximally selected statistic for small n

- **Target file(s).** `sigma/_splitting.py:31-210`, new
  `sigma/_max_rank_exact.py`.
- **Change.** Port Hothorn & Lausen's (2003) Algorithm 1 - the recursive
  enumeration of the linear-rank statistic distribution under all
  permutations of the rank vector - and use it for cutpoint p-values when
  `n <= n_exact_threshold` (default 30). Fall back to the existing
  Monte-Carlo approximation otherwise.
- **Success criterion.** A unit test checks the exact path matches the
  Monte-Carlo path within Monte-Carlo error on a 25-row dataset.
- **Paper(s) addressed.** 7.1 (YELLOW).

### 12. Cite the Lego and max-rank papers inline in the source

- **Target file(s).** `sigma/_statistics.py:1-7` (module docstring),
  `sigma/_splitting.py:1-7` (module docstring).
- **Change.** Add a one-line "References" block under each module's
  existing Hothorn-2006 citation:
  - In `_statistics.py`: cite Hothorn et al. (2006) Lego.
  - In `_splitting.py`: cite Hothorn & Lausen (2003).
- **Success criterion.** `grep -r "2003"` in `sigma/` returns the Hothorn
  & Lausen citation; `grep -r "2006.*Lego"` returns the Lego citation.
- **Paper(s) addressed.** 5.4 (YELLOW), 7.4 (YELLOW).

---

## Verification of this audit

Use this checklist to confirm the audit is sound before acting on it.

1. **Spot-check three random rows** by opening the cited paper PDF and
   the Sigma file at the named line. Confirm the recommendation is real
   (not paraphrased into something the paper does not say) and the code
   matches the row's quote and score.
2. **Confirm every YELLOW / RED row is referenced** in at least one
   action item ("paper(s) addressed").
3. **Re-read the website synthesis** for each paper and confirm the
   audit's "what Sigma claims" lines up with the website's claim. Where
   they diverge (notably paper 11), the action plan must propose either a
   code fix or a website correction.
4. **Run `mamba run -n standard python -m unittest discover tests`**
   after each action lands and confirm no regressions.
