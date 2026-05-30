# Sigma

<img src="https://arschitectura.com/medias/sigma_small.webp" alt="Sigma" width="200" height="200" align="right">

**Conditional inference trees for Python.**

Provides regression (`RegressionTree`), classification (`ClassificationTree`),
and survival-analysis (`SurvivalTree`) estimators, compatible with
scikit-learn.

- **Unbiased splits** - permutation-based p-values decouple variable selection from split search, avoiding CART's bias toward variables with many possible splits
- **Interpretable by construction** - each split is a statistical hypothesis test with a reported p-value, and fitted trees render to PNG/SVG via `to_image`
- **scikit-learn compatible** - `RegressionTree`, `ClassificationTree`, and `SurvivalTree` drop into any sklearn pipeline

Every statistical method in Sigma comes from a [peer-reviewed paper](#references).

## 1. License

Governed by the [**Sigma License**](./LICENSE.txt). This is a
source-available license, not OSI-approved open source. Commercial use
is permitted with attribution. ArsChitectura SAS retains an at-will right to
revoke the license, at any time, for any reason. Licensee shall consult
[Licensor's organization website](https://arschitectura.com/products/sigma/) and
the [Software's project homepage](https://github.com/arschitectura/sigma) at
least once every ninety (90) days.

**AI, ML, and other automated ingestion of this library, its
documentation, or any derivative work is prohibited**, excepted to
generate your own client code that calls Sigma's public API.

External contributors must sign the CLA in [`CONTRIBUTING.md`](./CONTRIBUTING.md)
before a pull request can be accepted.

If your needs exceeds this, a paid, non-revocable commercial license is available on [request](https://arschitectura.com/contact/).

## 2. Support

Read the [documentation](https://arschitectura.com/products/sigma/).

Have questions, feedback, or need help getting started? We'd love to hear from you - [get in touch](https://arschitectura.com/contact/).

<div align="center">
  <a href="https://arschitectura.com/contact/">
    <img src="https://arschitectura.com/medias/card.webp" alt="Card" width="500" height="311">
  </a>
</div>

## 3. Installation

```bash
pip install ars-sigma
```

## 4. Sample Trees

Three trees fitted on classic datasets. Each subsection shows the
fit code, the `to_text` rendering, and the rendered tree and response
images. Click an image to view it at full size.

### 4.1. Titanic (classification)

Predicting survival probability with a Jeffreys 95% confidence
interval at each leaf - surfaces passenger class, sex, and age.

```python
tree = sigma.ClassificationTree(random_state=123)
tree.fit(X, y)
print(tree.to_text(precision=1))
tree.to_image("png", "titanic.png", precision=1)
tree.to_image("png", "titanic_response.png", kind="response")
```

```
                                                 Died proba.        Survived proba. Obs. count Obs. share Split p-value Leaf index
                                      ---------------------- ---------------------- ---------- ---------- ------------- ----------
All records                           59.6% (55.9% to 63.1%) 40.4% (36.9% to 44.1%)        712     100.0%         0.02%
├── Passenger class is "1st" or "2nd" 43.1% (38.1% to 48.3%) 56.9% (51.7% to 61.9%)        357      50.1%         0.02%
│   ├── Sex is "female"                 5.7% (2.9% to 10.2%) 94.3% (89.8% to 97.1%)        157      22.1%                        6
│   └── Sex is "male"                 72.5% (66.0% to 78.3%) 27.5% (21.7% to 34.0%)        200      28.1%         0.08%
│       ├── Passenger class is "1st"  60.4% (50.7% to 69.5%) 39.6% (30.5% to 49.3%)        101      14.2%         1.02%
│       │   ├── Age <= 53.0           53.2% (42.2% to 63.9%) 46.8% (36.1% to 57.8%)         79      11.1%                        5
│       │   └── Age > 53.0            86.4% (67.9% to 96.0%)  13.6% (4.0% to 32.1%)         22       3.1%                        2
│       └── Passenger class is "2nd"  84.8% (76.8% to 90.9%)  15.2% (9.1% to 23.2%)         99      13.9%         0.62%
│           ├── Age <= 12.0                 0% (0% to 23.8%)   100% (76.2% to 100%)          9       1.3%                        7
│           └── Age > 12.0            93.3% (86.8% to 97.2%)   6.7% (2.8% to 13.2%)         90      12.6%                        1
└── Passenger class is "3rd"          76.1% (71.4% to 80.3%) 23.9% (19.7% to 28.6%)        355      49.9%         0.02%
    ├── Sex is "female"               53.9% (44.3% to 63.4%) 46.1% (36.6% to 55.7%)        102      14.3%                        4
    └── Sex is "male"                 85.0% (80.2% to 89.0%) 15.0% (11.0% to 19.8%)        253      35.5%                        3
```

<table>
  <tr>
    <td><a href="https://arschitectura.com/medias/sigma_titanic.png"><img src="https://arschitectura.com/medias/sigma_titanic.png" alt="Tree fitted on the Titanic dataset"></a></td>
  </tr>
  <tr>
    <td><a href="https://arschitectura.com/medias/sigma_titanic_response.png"><img src="https://arschitectura.com/medias/sigma_titanic_response.png" alt="Response plot for the Titanic dataset"></a></td>
  </tr>
</table>

### 4.2. Diabetes (regression)

Predicting one-year disease progression with a Bayesian-bootstrap 95%
confidence interval at each leaf - surfaces age, BMI, and HDL
cholesterol.

```python
tree = sigma.RegressionTree(random_state=123, reverse_order=True)
tree.fit(X, y)
print(tree.to_text(precision=1))
tree.to_image("png", "diabetes.png", orientation="left-to-right", precision=1)
tree.to_image("png", "diabetes_response.png", kind="response")
```

```
                                           Disease progression mean Obs. count Obs. share Split p-value Leaf index
                                           ------------------------ ---------- ---------- ------------- ----------
All records                                  152.1 (145.1 to 159.3)        442     100.0%         0.02%
├── Age <= 49                                136.7 (127.1 to 147.0)        214      48.4%         0.02%
│   ├── BMI <= 27.0                          108.7 (100.0 to 118.2)        143      32.4%         0.02%
│   │   ├── HDL cholesterol > 43.5             95.0 (86.7 to 104.5)        109      24.7%         2.92%
│   │   │   ├── Age > 24                       89.8 (80.9 to 100.3)         92      20.8%                       11
│   │   │   └── Age <= 24                    123.3 (104.0 to 144.2)         17       3.8%                        7
│   │   └── HDL cholesterol <= 43.5          152.5 (134.3 to 171.3)         34       7.7%         0.68%
│   │       ├── Total cholesterol <= 181     119.4 (100.5 to 139.1)         17       3.8%                        9
│   │       └── Total cholesterol > 181      185.7 (163.9 to 209.0)         17       3.8%                        4
│   └── BMI > 27.0                           193.1 (176.0 to 210.0)         71      16.1%         1.30%
│       ├── Blood sugar <= 103               179.6 (162.2 to 197.7)         60      13.6%                        5
│       └── Blood sugar > 103                267.1 (249.9 to 283.8)         11       2.5%                        1
└── Age > 49                                 166.6 (156.9 to 176.6)        228      51.6%         0.02%
    ├── BMI <= 26.9                          130.2 (119.4 to 141.4)        124      28.1%         0.02%
    │   ├── Triglycerides (log) <= 4.7        104.9 (95.5 to 115.4)         80      18.1%                       10
    │   └── Triglycerides (log) > 4.7        176.2 (157.9 to 194.1)         44      10.0%                        6
    └── BMI > 26.9                           210.0 (196.1 to 223.3)        104      23.5%         0.02%
        ├── Blood pressure <= 111.8          193.1 (178.3 to 207.8)         78      17.6%         1.64%
        │   ├── Triglycerides (log) <= 4.4    122.6 (98.5 to 144.7)         10       2.3%                        8
        │   └── Triglycerides (log) > 4.4    203.4 (188.4 to 218.9)         68      15.4%                        3
        └── Blood pressure > 111.8           260.8 (240.9 to 277.6)         26       5.9%                        2
```

<table>
  <tr>
    <td><a href="https://arschitectura.com/medias/sigma_diabetes.png"><img src="https://arschitectura.com/medias/sigma_diabetes.png" alt="Tree fitted on the Diabetes dataset"></a></td>
  </tr>
  <tr>
    <td><a href="https://arschitectura.com/medias/sigma_diabetes_response.png"><img src="https://arschitectura.com/medias/sigma_diabetes_response.png" alt="Response plot for the Diabetes dataset"></a></td>
  </tr>
</table>

### 4.3. GBSG-2 breast cancer (survival)

Predicting recurrence-free years with a Brookmeyer-Crowley 95%
confidence interval at each leaf - splits on positive lymph nodes,
hormone therapy, and progesterone receptor level.

```python
tree = sigma.SurvivalTree(
    random_state=123,
    metrics=("median", ("survival", 5.0, "years")),
)
tree.fit(X, y)
print(tree.to_text(precision=1))
tree.to_image("png", "breast_cancer.png", precision=1)
tree.to_image("png", "breast_cancer_response.png", kind="response")
```

```
                                              Median Recurrence-free years    Survival at 5 years Obs. count Obs. share Split p-value Leaf index
                                              ---------------------------- ---------------------- ---------- ---------- ------------- ----------
All records                                               4.9 (4.2 to 5.5) 49.2% (44.6% to 53.6%)        686     100.0%         0.02%
├── Tumor size <= 19                              unknown (5.4 to unknown) 65.6% (55.4% to 73.9%)        135      19.7%         0.04%
│   ├── Positive lymph nodes <= 2                 unknown (unknown bounds) 80.7% (68.2% to 88.7%)         77      11.2%                        7
│   └── Positive lymph nodes > 2                          4.7 (2.6 to 5.5) 44.7% (29.2% to 59.1%)         58       8.5%         3.52%
│       ├── Age > 41                                  5.4 (3.5 to unknown) 54.6% (36.2% to 69.8%)         49       7.1%                        5
│       └── Age <= 41                                     1.3 (0.7 to 3.2)          0% (0% to 0%)          9       1.3%                        1
└── Tumor size > 19                                       4.3 (3.7 to 5.0) 44.9% (39.7% to 49.9%)        551      80.3%         0.02%
    ├── Positive lymph nodes <= 4                     5.7 (4.8 to unknown) 54.7% (47.7% to 61.0%)        332      48.4%         1.10%
    │   ├── Hormone therapy is true               unknown (5.6 to unknown) 67.7% (56.3% to 76.7%)        114      16.6%                        6
    │   └── Hormone therapy is false                  4.8 (3.9 to unknown) 47.1% (38.3% to 55.4%)        218      31.8%                        4
    └── Positive lymph nodes > 4                          2.4 (2.0 to 3.1) 29.9% (22.6% to 37.5%)        219      31.9%         0.10%
        ├── Progesterone receptor level > 24              4.1 (2.7 to 5.5) 43.8% (31.4% to 55.4%)        107      15.6%                        3
        └── Progesterone receptor level <= 24             1.7 (1.4 to 2.2) 17.3% (10.0% to 26.3%)        112      16.3%                        2
```

<table>
  <tr>
    <td><a href="https://arschitectura.com/medias/sigma_breast_cancer.png"><img src="https://arschitectura.com/medias/sigma_breast_cancer.png" alt="Tree fitted on the GBSG-2 breast cancer dataset"></a></td>
  </tr>
  <tr>
    <td><a href="https://arschitectura.com/medias/sigma_breast_cancer_response.png"><img src="https://arschitectura.com/medias/sigma_breast_cancer_response.png" alt="Response plot for the GBSG-2 breast cancer dataset"></a></td>
  </tr>
</table>

## 5. Advanced usage

### 5.1. Controlling tree depth and node size

`alpha` is the principal knob: it sets the significance threshold for
every split test, so lowering it produces a terser, more statistically
conservative tree, and raising it produces a richer, more exploratory
one. `min_splits`, `min_buckets`, and `max_depth` are secondary safety
bounds, shared between `RegressionTree` and `ClassificationTree`.

```python
tree = ClassificationTree(
    max_depth=4,            # maximum tree depth (None = unlimited)
)
```

### 5.2. Fitting with sample weights

Sample weights let you model **variable exposures** - per-row
time-at-risk, insurance policy-years, or frequency weights for
pre-aggregated rows. A weight of `k` is equivalent to observing the
sample `k` times.

```python
import numpy
from sigma import RegressionTree

n = 200
X = numpy.random.randn(n, 2)
claim_amount = numpy.where(X[:, 0] > 0, 1500.0, 300.0) + 100 * numpy.random.randn(n)
exposure_years = numpy.random.uniform(0.1, 2.0, size=n)

tree = RegressionTree()
tree.fit(X, claim_amount, sample_weight=exposure_years)
predictions = tree.predict(X)
```

### 5.3. Visualizing the tree

Install the optional visualization extra and the Graphviz system
binary (`brew install graphviz` on macOS):

```bash
pip install ars-sigma[viz]
```

Then render to PNG, PDF, SVG, or GIF:

```python
tree.to_image("png", "tree.png", feature_names=["feature_1", "feature_2"], response_name="y")
```

PNG and PDF additionally require `cairosvg`; SVG needs only the
Graphviz binary. See `to_image` and `export_graphviz` for the full set
of display options.

### 5.4. Exporting the tree as a SQL CASE expression

`to_sql` (and the module-level `sigma.export_sql`) emits a single SQL
`CASE` expression that reproduces `tree.predict` row-by-row in any
SQL-92/SQL-99 engine, with no extra dependencies:

```python
sql_expression = tree.to_sql()
print(sql_expression)
# SELECT id, (<sql_expression>) AS prediction FROM points;
```

```sql
CASE
    WHEN "Passenger class" IN ('1st', '2nd') THEN
        CASE
            WHEN "Sex" = 'female' THEN
                0.9426751592356688 -- Leaf 6
            WHEN "Sex" = 'male' THEN
                CASE
                    WHEN "Passenger class" = '1st' THEN
                        CASE
                            WHEN "Age" <= 53.0 THEN
                                0.46835443037974683 -- Leaf 5
                            WHEN "Age" > 53.0 THEN
                                0.13636363636363635 -- Leaf 2
                            ELSE NULL
                        END
                    WHEN "Passenger class" = '2nd' THEN
                        CASE
                            WHEN "Age" <= 12.0 THEN
                                1.0 -- Leaf 7
                            WHEN "Age" > 12.0 THEN
                                0.06666666666666667 -- Leaf 1
                            ELSE NULL
                        END
                    ELSE 0.275
                END
            ELSE 0.5686274509803921
        END
    WHEN "Passenger class" = '3rd' THEN
        CASE
            WHEN "Sex" = 'female' THEN
                0.46078431372549017 -- Leaf 4
            WHEN "Sex" = 'male' THEN
                0.15019762845849802 -- Leaf 3
            ELSE 0.23943661971830985
        END
    ELSE 0.4044943820224719
END
```

For `ClassificationTree`, pass `target_class=` to pick which class
probability the expression should emit. Categorical values not seen at
fit time evaluate to the holding node's prediction, mirroring
`tree.predict`. `NULL` numerical or boolean inputs fall through to
`ELSE NULL`; wrap in `COALESCE(..., default)` to substitute a fallback
value.

## 6. Parameters

The table below is a quick reference; each parameter has a dedicated
subsection further down with defaults, alternatives, and guidance on
when to choose each option.

| Parameter                         | Description                                                                  |
| :-------------------------------- | :--------------------------------------------------------------------------- |
| `correlation`                     | Rank-transform inputs (robust) or use raw values (classical)                 |
| `test_stat`                       | How the multivariate score is aggregated into a scalar test statistic        |
| `test_type`                       | Multiplicity adjustment applied across covariates                            |
| `alpha`                           | Significance level for the stopping rule                                     |
| `min_splits`                      | Minimum sum of weights required to attempt a split                           |
| `min_buckets`                     | Minimum sum of weights in each child node                                    |
| `max_depth`                       | Maximum tree depth                                                           |
| `categorical_features`            | Which feature columns are categorical                                        |
| `ci_method` (regression tree)     | Confidence interval method for node mean predictions                         |
| `ci_method` (classification tree) | Confidence interval method for per-class proportions                         |
| `ci_coverage`                     | Coverage level for node-prediction confidence intervals                      |
| `transmuter`                      | Per-node data transform with post-hoc split validation                       |
| `resamples`                       | Number of permutations for `test_type="monte_carlo"`                         |
| `decorator`                       | Per-node decoration callable rendered by `to_text` / `to_image`              |
| `random_state`                    | RNG seed for permutation resampling, bootstrap CI methods, and plot jitter   |

### 6.1. `correlation`

**Default**: `"rank"`.

Score function for the test statistic.

- `"normal"` uses raw values, recovering the original Pearson-like
  behavior from Hothorn et al. (2006). Choose this when the response is
  well-behaved (approximately Gaussian, no heavy outliers) and you want
  the slight power gain on truly linear associations.
- `"rank"` (default) rank-transforms continuous covariates and, for
  regression, the response before computing the statistic, yielding a
  Spearman-like nonparametric test. Robust to outliers and heavy tails.
  The safe choice for arbitrary real-world data.

### 6.2. `test_stat`

**Default**: `"quadratic"`.

How the multivariate score is aggregated into a scalar test statistic.

- `"maximum"` is a maximum-type statistic that concentrates power on
  alternatives where one component dominates. Choose this when you
  expect a single-direction effect (e.g., a binary classification where
  only one class differs from the rest).
- `"quadratic"` (default) is an omnibus chi-squared form with good
  power across general alternatives. Choose this when you have no prior
  on the direction of association, or when the response is multivariate
  (multi-class classification with many classes).

### 6.3. `test_type`

**Default**: `"sidak"`.

Multiplicity adjustment applied across the $m$ candidate covariates
(indexed by $j$), transforming each raw p-value $P_j$ before the
stopping rule fires.

- `"bonferroni"` is the closed-form $\min(m P_j, 1)$. The simplest and
  best-known correction, strictly more conservative than Sidak under
  independence; prefer `"sidak"` unless matching an external reference.
- `"monte_carlo"` is the Westfall-Young min-P resampling procedure.
  More powerful than Sidak when covariates are correlated, at the cost
  of $B \cdot m$ extra statistic evaluations per node, where $B$ is the
  number of response permutations (controlled by `resamples`). Choose
  when covariates are highly collinear and you can afford the
  resampling budget; requires a positive `resamples`.
- `"sidak"` (default) is the closed-form $1 - (1 - P_j)^m$. Powerful
  under independence or positive dependence of test statistics. The
  recommended default.

### 6.4. `alpha`

**Default**: `0.05`.

Significance level for the stopping rule. Recursion stops at a node
when $\min_j(\text{adjusted } P_j) > \alpha$.

The default `0.05` is a good choice for simple, exploratory analysis.
For trees fitted on very large datasets, or on correlated records
where the independence assumption is partially broken, tighten
`alpha` by one or two orders of magnitude (`0.005` or `0.0005`) to
keep the tree compact. For models aiming at higher predictive
accuracy (closer to a full-fledged machine learning model), loosen
`alpha` to between `0.10` and `0.25`. Tune in concert with
`max_depth`, `min_splits`, and `min_buckets`.

### 6.5. `min_splits`

**Default**: `20`.

Minimum sum of weights required to attempt a split. Nodes whose weight
sum falls below this become leaves regardless of p-values. Increase to
enforce statistical reliability of node-level estimates on smaller
subsets, decrease to allow finer partitioning.

### 6.6. `min_buckets`

**Default**: `7`.

Minimum sum of weights in each child node. Splits that would produce a
child smaller than this are rejected. Together with `min_splits`,
controls the smallest leaf permitted; raise both for noisier data.

### 6.7. `max_depth`

**Default**: `None` (no limit).

Maximum tree depth. Set to a small integer for shallow, easily interpreted
trees. Leave `None` to let the p-value stopping rule fully control depth.

### 6.8. `categorical_features`

**Default**: `None` (all numeric).

List of feature columns to treat as categorical. Entries may be
column-name strings (resolved against the DataFrame columns at fit
time, i.e. `feature_names_in_`) or integer column indices; mixing the
two forms is allowed. Letting $K$ denote the number of levels in a
categorical feature, Sigma uses exhaustive split enumeration for
$K \le 10$ and an ordered-merge heuristic for $K > 10$ (see the
Algorithm section).

### 6.9. `ci_method` (`RegressionTree` only)

**Default**: `"bayesian_bootstrap"`.

Method for the confidence interval on each leaf's mean prediction. In
the descriptions below, $y$ denotes the per-row response, $n$ the
sample size at the leaf, and $n_{\text{eff}}$ the Kish effective
sample size at the leaf.

- `"bayesian_bootstrap"` (default) uses Dirichlet resampling of the
  weighted mean. Nonparametric: makes no assumption on the response
  distribution. The safe choice for arbitrary regression targets, but
  less powerful than a method tailored to the response's actual
  distribution.
- `"bca"` is the bias-corrected and accelerated bootstrap interval
  (Efron, 1987): resample $10{,}000$ times from the empirical
  distribution, then read percentiles corrected for median
  bias ($z_0$) and skewness ($a$, computed via jackknife).
  Nonparametric and second-order accurate ($O(1/n)$ coverage error);
  transformation-respecting. Choose for the frequentist counterpart of
  `"bayesian_bootstrap"` when an external benchmark specifies a
  frequentist confidence interval. Non-deterministic across calls.
- `"beta"` is a Clopper-Pearson-style Beta interval for proportional
  responses in $[0, 1]$. Choose when $y$ is naturally a rate or
  proportion (conversion rate, click-through rate).
- `"exponential"` is the exact chi-squared interval for an Exponential
  mean (Gamma with shape $= 1$); requires $y \ge 0$. Choose when
  responses are non-negative waiting times or lifetimes that follow an
  exponential distribution.
- `"gamma"` is the exact chi-squared interval for a Gamma mean using
  a method-of-moments shape estimate; requires $y \ge 0$. Choose for
  non-negative right-skewed responses (insurance claims, incomes,
  durations).
- `"log_normal"` is Cox's interval for the arithmetic mean of a
  log-normal response; requires $y > 0$. Centered on the log-normal
  MLE of the mean, not the sample mean. Choose when $\log y$ is
  approximately normal (financial returns, biological measurements).
- `"log_normal_gci"` is the generalized confidence interval
  (Krishnamoorthy & Mathew, 2003) for the arithmetic mean of a
  log-normal response; requires $y > 0$. Like `"log_normal"` but built
  via Monte Carlo from a generalized pivot, giving asymmetric bounds.
  Choose when $n_{\text{eff}}$ is very small with large $\log y$ variance, where
  Cox's symmetric Wald form begins to lose calibration.
  Non-deterministic across calls.
- `"normal"` is a Wald-style interval $\bar{Y} \pm z \cdot \text{SE}$
  ($\bar{Y}$ the leaf weighted mean of $y$, $z$ a standard normal
  quantile, $\text{SE}$ the standard error) with the Kish effective
  sample size. Tight and cheap. Choose when the central limit theorem
  applies comfortably ($n_{\text{eff}}$ well above 30, finite response
  variance).
- `"poisson"` is the exact Garwood chi-squared interval for a Poisson
  mean rate; requires $y \ge 0$. The conservative choice with
  guaranteed coverage (Patil & Kulkarni, 2012). Choose for count
  responses generated by an approximately Poisson process when
  guaranteed coverage matters more than tightness.
- `"poisson_jeffreys"` is the equal-tailed Jeffreys interval for a
  Poisson mean rate; requires $y \ge 0$. Shorter than `"poisson"` at
  moderate rates (Patil & Kulkarni, 2012). Choose for count
  responses at moderate rates when you do not require Garwood's
  guaranteed coverage.
- `"student_t"` has the same form as `"normal"` but uses a Student-t
  quantile with $n_{\text{eff}} - 1$ degrees of freedom. Wider than
  `"normal"` for small effective sample sizes. Choose when
  $n_{\text{eff}}$ is borderline and small-sample coverage matters.

### 6.10. `ci_method` (`ClassificationTree` only)

**Default**: `"jeffreys"`.

Method for the per-class confidence intervals on leaf class
proportions. In the descriptions below, $n$ denotes the leaf sample
size and $z$ a standard normal quantile.

- `"agresti_coull"` is the adjusted Wald interval: Wald applied
  after adding $z^2/2$ pseudo-successes and $z^2/2$ pseudo-failures.
  Slightly wider and more conservative than `"wilson"` at small
  sample sizes; statistically equivalent to `"wilson"` and
  `"jeffreys"` for $n > 40$ per Brown-Cai-DasGupta (2001). Choose
  when matching an external reference that specifies Agresti-Coull.
- `"clopper_pearson"` is the exact Beta interval. Has the absolute
  coverage *guarantee* ($\ge$ `ci_coverage` for every true proportion),
  but is conservative: intervals are wider than they need to be on
  average. Choose when guaranteed coverage matters more than tightness
  (regulatory or safety contexts).
- `"jeffreys"` (default) is a Bayesian interval from the Beta
  posterior with the Jeffreys non-informative prior $\mathrm{Beta}(0.5, 0.5)$.
  Neither systematically conservative nor systematically aggressive on
  average. Recommended for general use.
- `"mid_p_exact"` is the mid-p variant of Clopper-Pearson. Strictly
  narrower than `"clopper_pearson"` while keeping an exact-tail
  rationale, with average coverage close to nominal. Choose when
  Clopper-Pearson's conservatism feels too wasteful but an exact-tail
  method is still desired.
- `"wilson"` is the closed-form Wilson score interval, clipped to
  $[0, 1]$. Cheapest to compute and accurate at moderate sample sizes;
  coverage degrades near 0 and 1. Choose when you need vectorized
  speed and class proportions are not extreme.
- `"wilson_cc"` is the Wilson score interval with Newcombe's
  continuity correction. Slightly wider than `"wilson"`, restoring
  lower-tail coverage at small sample sizes. Choose when the leaf
  total weight $w_{\text{total}}$ is small and plain Wilson
  under-covers.

### 6.11. `ci_coverage`

**Default**: `0.95`.

Coverage level for node-prediction confidence intervals. Set to `None`
to skip CI computation entirely (the proper way to fully avoid the
per-node `ci_method` cost). Common alternatives: `0.90` (less
conservative), `0.99` (more conservative). For survival trees, also
controls the confidence band drawn behind each Kaplan-Meier curve in
the response plot.

### 6.12. `transmuter`

**Default**: `None`.

Optional callable that transforms node-level data before predictions
and confidence intervals are computed, with post-hoc split validation.
Signature: `(X, y, sample_weight) -> (y', sample_weight')`, or
`(X, y, sample_weight, side_data) -> (y', sample_weight')` when
`side_data` is passed to `fit`. After each candidate split, both
child subsets are independently transmuted and a significance test is
run on the transmuted data; if the p-value exceeds `alpha` the split
is rejected and the node becomes a leaf. Use cases: survival outcomes
(Kaplan-Meier-style transformation), rate normalization (impressions
to click-through rate), de-noising heavy-tailed responses.

### 6.13. `resamples`

**Default**: `None`.

Number of permutations $B$ for `test_type="monte_carlo"`. Required and
must be a positive integer when monte_carlo is selected; ignored
otherwise. Typical choices: `1000` for day-to-day production, `10000`
for paper-grade reproducible adjusted p-values.

### 6.14. `decorator`

**Default**: `None`.

Optional callable invoked once per node after the tree is built.
Signature: `(X_active, y_active, w_active, side_data_active) ->
decoration` where `decoration` is any object (or `None`). The returned
object is stored on the node as `node.decoration` and rendered by
`to_text` and `to_image`. Use cases: per-leaf metric (RMSE,
classification accuracy), business labels (segment names), diagnostic
statistics.

### 6.15. `random_state`

**Default**: `None`.

Seed for all stochastic operations in the estimator. Pass an integer
for reproducibility; `None` uses an unpredictable seed. Controls:

- min-P permutation resampling under `test_type="monte_carlo"`;
- the bootstrap-family CI methods of `RegressionTree`
  (`bayesian_bootstrap`, `bca`, `log_normal_gci`);
- the jitter of `to_image(kind="response")` raincloud plots
  (`RegressionTree` only; combined with the leaf index so each leaf
  receives a distinct pattern).

## 7. Algorithm

The algorithm builds a decision tree using statistical hypothesis
testing for unbiased variable selection. Unlike CART, which selects
variables by maximizing an impurity criterion (and is therefore biased
toward variables with many possible splits), conditional inference trees
use permutation-based p-values to decouple variable selection from split
search.

The framework is generic: the only difference between regression and
classification is the influence function $h$ applied to the response
$Y_i$ of observation $i$. For regression, $h(Y_i) = Y_i$ (identity).
For classification with $J$ classes, $h(Y_i) = e_J(Y_i)$ (one-hot
encoding of the class label). All test statistics, p-value
computations, and splitting criteria use the same formulas.

### 7.1. Step 1: Variable selection and stopping

Given $n$ observations with response values $Y_i$, covariate values
$X_{ji}$ (the value of the $j$-th covariate $X_j$ for observation $i$),
and case weights $w_i$, define $g_j$ as the score function for
covariate $X_j$ (identity for numeric covariates, dummy encoding for
categorical ones). When
`correlation="rank"` (the default), continuous covariates and regression
responses are rank-transformed within each node before computing the
test statistics, yielding Spearman-like nonparametric tests that are
robust to outliers and non-normality. When `correlation="normal"`, raw
values are used (Pearson-like, as in the original paper). For each
covariate $X_j$, the algorithm computes the linear statistic

$$T_j = \text{vec}\!\left(\sum_{i=1}^{n} w_i \cdot g_j(X_{ji}) \cdot h(Y_i)^\top\right)$$

and derives its conditional expectation $\mu_j$ and covariance
$\Sigma_j$ under the null hypothesis of independence between $X_j$ and
the response $Y$. A test statistic (quadratic-form or maximum-type) is
computed and converted to a p-value $P_j$. A multiplicity adjustment is
applied across all $m$ covariates, and recursion stops when
$\min_j(\text{adjusted } P_j) > \alpha$. Otherwise the covariate with
the smallest adjusted p-value is selected.

The default adjustment is the Sidak correction
($\text{adjusted } P_j = 1 - (1 - P_j)^m$), which is powerful under the
mild assumption that the test statistics across covariates are
independent or positively dependent. A simpler closed-form alternative,
`test_type="bonferroni"`, uses
$\text{adjusted } P_j = \min(m P_j, 1)$; it is strictly more
conservative than Sidak. The third alternative,
`test_type="monte_carlo"`, uses the Westfall-Young (1993) min-P
resampling procedure. For each of $B$ permutations of the response, all
$m$ p-values are recomputed and the minimum recorded. The adjusted
p-value for covariate $j$ is the proportion of permutations where this
minimum did not exceed the observed $P_j$. This method is more powerful
than Sidak when covariates are correlated, at the cost of
$O(B \cdot m)$ additional statistic evaluations. Set `resamples` (e.g.,
1000 or 10000) and optionally `random_state` for reproducibility. All
three methods are available via the `test_type` parameter.

### 7.2. Step 2: Binary splitting

For the selected covariate, the algorithm searches for the binary
partition $A^*$ that maximizes the two-sample test statistic. Numeric
covariates are split at midpoints between consecutive unique values.
Categorical covariates with $K \le 10$ levels use exhaustive enumeration
of all $2^{K-1} - 1$ partitions; for $K > 10$, categories are ordered
by weighted mean of the first influence function column and only $K - 1$
contiguous splits are evaluated (provably optimal for regression,
heuristic for classification).

### 7.3. Step 3: Recursion and prediction

Case weights are updated to reflect node membership and steps 1-2 are
repeated recursively on each child node. Terminal nodes predict:

- **Regression**: the weighted mean of the response.
- **Classification**: the majority class, with class probabilities
  given by the normalized weighted class counts.

## 8. Partykit compatibility

Sigma is a pure-Python reimplementation of R's `partykit::ctree` with
various improvements. Tree shape, split variables, split thresholds,
and per-leaf predictions are empirically verified to match
`partykit::ctree` on three reference datasets, one per task family:

- **Regression**: the `airquality` dataset (Ozone on Wind/Temp/Month/Day,
  n=116 after dropping the rows with no Ozone observation). Crosscheck at
  `tests/test_partykit_equivalence.py:26`.
- **Classification**: the `GlaucomaM` dataset from R's `TH.data` package
  (Class on 62 morphology covariates, n=196). Crosscheck at
  `tests/test_partykit_equivalence.py:75`.
- **Survival**: the `GBSG2` dataset from `lifelines`
  (`Surv(time, cens) ~ horTh + age + menostat + tsize + tgrade + pnodes +
  progrec + estrec`, n=686). Crosscheck at
  `tests/test_tree_survival.py:661`.

Three deliberate deviations from partykit are worth knowing about:

1. **`test_type="sidak"` is the default**, matching partykit's effective
   behavior. Partykit's `testtype="Bonferroni"` is a naming error on their
   part: the adjustment it computes is mathematically the Sidak formula
   $1 - (1 - P_j)^m$, not the textbook Bonferroni $\min(m P_j, 1)$.
   Sigma exposes both options under their correct names; pass
   `test_type="bonferroni"` for the textbook Bonferroni formula, or `test_type="sidak"`
   (the default) to match partykit's "Bonferroni" output exactly.
2. **`correlation="rank"` is the default**, where partykit uses raw
   values. Rank-transforming both response and continuous covariates
   gives a Spearman-style test that is robust to outliers and skew, at
   the cost of a small loss of power against linear alternatives. Pass
   `correlation="normal"` to match partykit exactly.
3. **Leaves are reordered for display**: `leaves_` iterates in a
   task-appropriate canonical order, and `to_text` / `to_image` swap
   left and right children of each inner node to match. Sort keys are
   ascending predicted response (`RegressionTree`), descending majority
   class share (`ClassificationTree`), and worst prognosis first
   (`SurvivalTree`). Partykit prints leaves in tree-traversal order.
   The underlying tree is identical; only the iteration order of
   `leaves_` and the visual left-vs-right placement of children in
   exported renderings differ.

## 9. References

- Hothorn, T., & Zeileis, A. (2015). *partykit: A Modular Toolkit for
  Recursive Partytioning in R.* *Journal of Machine Learning
  Research*, 16, 3905-3909.
  [jmlr.org/papers/v16/hothorn15a](https://jmlr.org/papers/v16/hothorn15a.html)
- Patil, V. V., & Kulkarni, H. V. (2012). *Comparison of Confidence
  Intervals for the Poisson Mean: Some New Aspects.* *REVSTAT -
  Statistical Journal*, 10(2), 211-227.
  [doi:10.57805/revstat.v10i2.117](https://doi.org/10.57805/revstat.v10i2.117)
- Hothorn, T., Hornik, K., & Zeileis, A. (2006). *Unbiased Recursive
  Partitioning: A Conditional Inference Framework.* *Journal of
  Computational and Graphical Statistics*, 15(3), 651-674.
  [doi:10.1198/106186006X133933](https://doi.org/10.1198/106186006X133933)
- Hothorn, T., Hornik, K., van de Wiel, M. A., & Zeileis, A. (2006).
  *A Lego System for Conditional Inference.* *The American
  Statistician*, 60(3), 257-263.
  [doi:10.1198/000313006X118430](https://doi.org/10.1198/000313006X118430)
- Olsson, U. (2005). *Confidence Intervals for the Mean of a
  Log-Normal Distribution.* *Journal of Statistics Education*, 13(1).
  [doi:10.1080/10691898.2005.11910638](https://doi.org/10.1080/10691898.2005.11910638)
- Krishnamoorthy, K., & Mathew, T. (2003). *Inferences on the Means of
  Lognormal Distributions Using Generalized p-Values and Generalized
  Confidence Intervals.* *Journal of Statistical Planning and
  Inference*, 115(1), 103-121.
  [doi:10.1016/S0378-3758(02)00153-2](https://doi.org/10.1016/S0378-3758\(02\)00153-2)
- Hothorn, T., & Lausen, B. (2003). *On the Exact Distribution of
  Maximally Selected Rank Statistics.* *Computational Statistics &
  Data Analysis*, 43(2), 121-137.
  [doi:10.1016/S0167-9473(02)00225-6](https://doi.org/10.1016/S0167-9473\(02\)00225-6)
- Brown, L. D., Cai, T. T., & DasGupta, A. (2001). *Interval
  Estimation for a Binomial Proportion.* *Statistical Science*, 16(2),
  101-133.
  [doi:10.1214/ss/1009213286](https://doi.org/10.1214/ss/1009213286)
- Agresti, A., & Coull, B. A. (1998). *Approximate is Better than
  "Exact" for Interval Estimation of Binomial Proportions.* *The
  American Statistician*, 52(2), 119-126.
  [doi:10.1080/00031305.1998.10480550](https://doi.org/10.1080/00031305.1998.10480550)
- Newcombe, R. G. (1998). *Two-Sided Confidence Intervals for the
  Single Proportion: Comparison of Seven Methods.* *Statistics in
  Medicine*, 17(8), 857-872.
  [doi:10.1002/sim.777](https://doi.org/10.1002/\(SICI\)1097-0258\(19980430\)17:8%3C857::AID-SIM777%3E3.0.CO;2-E)
- Efron, B. (1987). *Better Bootstrap Confidence Intervals.* *Journal
  of the American Statistical Association*, 82(397), 171-185.
  [doi:10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410)
- Rubin, D. B. (1981). *The Bayesian Bootstrap.* *Annals of
  Statistics*, 9(1), 130-134.
  [doi:10.1214/aos/1176345338](https://doi.org/10.1214/aos/1176345338)
- Wilson, E. B. (1927). *Probable Inference, the Law of Succession,
  and Statistical Inference.* *Journal of the American Statistical
  Association*, 22(158), 209-212.
  [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)
