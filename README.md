# Bayesian IPO study

Can an IPO's first trading days improve forecasts of its remaining
benchmark-adjusted return through trading day 60?

Start with **[analysis.ipynb](analysis.ipynb)**. It is the entry point for the
research question, data audit, model comparisons, uncertainty checks and worked
forecast. This repository is self-contained: it does not import code or read
research artifacts from the parent folder.

The reference snapshot is 7 September 2026. It contains 1,382 audited listings,
1,340 mature population candidates and 903 usable IPOs. The two development
blocks contain 298 validation observations in total; the final 2025–2026 cohort
contains 244 IPOs. The final $1bn+ and $2bn+ groups contain only 18 and 4 IPOs.

## 1. Reading order and project structure

For a first look, read `final_output.ipynb`, then `pycode/features.py`, then the
predictive functions in `pycode/bayesian_models.py`. Read `pycode/evaluation.py`
to understand how performance and uncertainty are assessed. The execution
scripts in `pycode/` are supporting tools, not additional research chapters.
Dependency files are in `requirements/`.

```text
Bayesian-Ipo-Prediction-Model/
├── README.md
├── final_output.ipynb
├── .gitignore
├── data/                 Frozen prepared inputs and experiment plan
├── models/               Saved numeric posterior and predictive draws
├── pycode/               Analysis code and execution scripts
│   ├── features.py
│   ├── bayesian_models.py
│   ├── evaluation.py
│   ├── check.py
│   ├── reproduce.py
│   └── refit.py
├── requirements/         Dependency files
│   ├── requirements.txt
│   └── requirements_fit.txt
└── results/              Reference estimates used by the notebook
```

There is no installable project package, configuration framework, encoded data
block or JSON sidecar. Data and results are CSVs. Model arrays are NumPy
`.npz` files loaded with `allow_pickle=False`. The notebook's own `.ipynb`
format is JSON internally, but the analysis code does not parse JSON.

### Files at the repository root

| File | Purpose | Reads or writes |
|---|---|---|
| `README.md` | This guide: setup, workflow, file catalogue and research limitations. | Documentation only. |
| `analysis.ipynb` | The research narrative, exploratory calculations, 18 figures, 20 displayed tables and the illustrative NEWCO forecast. It preserves the source notebook's 49 cells, including all 30 markdown cells. | Reads all three research folders. Running its cells does not save data or fitted models. |
| `features.py` | Builds early trading signals, the remaining-return target and chronological train/test splits. | Operates on supplied pandas tables; no file writes. |
| `bayesian_models.py` | Defines predictors, training-only standardisation, Bayesian fitting, model loading, predictive distributions and new-IPO forecasts. | Its loader reads a supplied model file. Its calculations return results to the caller rather than saving them. |
| `evaluation.py` | Aggregates scores; bootstraps paired model gains; evaluates size groups; calculates activity trajectories, decay fits and missing-outcome bounds. | Operates on supplied tables; no research-file writes. |
| `check.py` | Checks sample uniqueness, chronological cohorts, selected-model forecasts and future-suffix invariance against the reference snapshot. | Reads prepared data, saved models and two reference result tables. Does not write research artifacts. |
| `reproduce.py` | Clears notebook outputs in memory, executes all cells in a fresh Jupyter kernel and saves the executed notebook. | Reads and overwrites `analysis.ipynb` only after successful execution. |
| `refit.py` | Fits the 86 documented predictive experiments, calculates scores and comparisons, regenerates example forecasts and recalculates the two robustness exercises. | Reads prepared data and the reference example issuer list. Writes only beneath `results/refitted/`. |
| `requirements.txt` | Pins the direct dependencies used to run and check the notebook. | Installation instructions for the analysis environment. |
| `requirements_fit.txt` | Includes the analysis requirements and adds the Bayesian fitting dependencies. | Installation instructions for refitting. |
| `.gitignore` | Excludes virtual environments, bytecode, notebook checkpoints, Finder metadata and fresh refit outputs from version control. | Git configuration; it does not delete files. |

`.git/` contains Git-managed repository metadata, not research code.
If `.DS_Store` is present, it is macOS Finder metadata and has no role in the
analysis. Neither belongs in the scientific reading order.

### What the calculation files contain

**`features.py`**

- `early_features()`: uses only sessions 1 through the chosen cutoff to compute
  early excess return, cumulative trading intensity, realised volatility,
  log proceeds, drawdown and the feature date.
- `remaining_return()`: calculates the realised benchmark-adjusted return
  between the cutoff close and session 60. This is the target, not a predictor.
- `make_cutoff_data()`: assembles one row per usable IPO for a cutoff and
  benchmark, combining listing metadata, predictors and the target.
- `chronological_split()`: admits training IPOs only when both their listing
  and outcome dates precede the validation period.

**`bayesian_models.py`**

- `predictor_names()` and `standardize()`: define each specification's
  predictors and apply training-set means and scales.
- `load_model()`: reads numeric arrays and scalar metadata from a NumPy archive.
- `fit_model()`: fits a Gaussian or Student-t regression with PyMC and checks
  sampling diagnostics. PyMC and ArviZ are imported only inside this function.
- `predictive_parameters()` and `predictive_cdf()`: evaluate conditional
  locations, scales and distribution probabilities for posterior draws.
- `predict()`: returns probabilities, predictive quantiles and, when outcomes
  are supplied, log scores, CRPS, PIT values and interval-coverage indicators.
- `forecast()`: applies a frozen model to offering information and an observed
  trading prefix. It slices the input before checking it, so future suffixes
  cannot affect the forecast.

**`evaluation.py`**

- `score_table()`: averages forecast scores and calculates interval coverage
  with Jeffreys binomial intervals.
- `bootstrap_gain()`: resamples matched IPO-level score differences to estimate
  a model's incremental log-score gain and uncertainty.
- `model_comparisons()`: applies that paired comparison across models, cutoffs,
  phases and offering-size thresholds.
- `size_performance()`: aggregates predictive performance within size groups.
- `missing_outcome_bounds()`: combines observed event rates, Jeffreys sampling
  intervals and worst-case assumptions about missing outcomes.
- `activity_trajectories()`: computes daily and cumulative activity, including
  retrospective ratios relative to each IPO's sessions 41–60.
- `activity_distributions()`: calculates event-day quantiles for all usable
  IPOs and the $1bn+ group.
- `fit_activity_decay()`: compares constant, power-law and exponential OLS
  descriptions, bootstrapping complete IPO trajectories for slope uncertainty.

## 2. Run the notebook

The tested interpreter is Python 3.13. The scientific analysis uses NumPy,
pandas, Matplotlib and SciPy. SciPy provides Student-t probabilities, density
evaluation and beta-distribution intervals. Scikit-learn is **not currently
used**. Jupyter-related packages handle notebook execution, not modelling.

From this repository's folder on macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name "IPO study"
jupyter lab analysis.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate` instead
of the `source` command.

Choose the **IPO study** kernel and run all cells. Launch Jupyter from this
folder because the notebook deliberately uses simple relative paths.

The kernel registration above is important: `reproduce.py` explicitly requests
the kernel named `python3`. Registering that name inside the active environment
avoids accidentally running the notebook with an unrelated global interpreter.
The notebook kernel's display name and its internal name are different things.

For automatic execution and numerical checks:

```bash
python check.py
python reproduce.py
```

`check.py` uses ordinary Python assertions as well as NumPy comparisons;
run it normally, not with Python's `-O` flag. Its fixed cohort counts and
reference forecasts test this particular snapshot, not arbitrary new datasets.
Initial package installation needs internet access; subsequent notebook
execution does not download market data.

## 3. Refit the predictive models

Refitting requires PyMC, PyTensor, ArviZ and xarray in addition to the analysis
environment:

```bash
python -m pip install -r requirements_fit.txt
python refit.py
```

The script uses four chains, 800 tuning steps and 800 retained draws per chain.
The random seed is `20260908`. Predictors and prior scales are calculated
within each training sample. A fit is rejected if reported maximum R-hat exceeds
1.01, minimum bulk or tail ESS is below 400, any divergences occur, or minimum
BFMI is at most 0.3.

On macOS, if installed Apple compiler tools report `vector file not found`,
the following command supplies the SDK header location:

```bash
CPLUS_INCLUDE_PATH="$(xcrun --show-sdk-path)/usr/include/c++/v1" python refit.py
```

A compiler-free alternative exists but can be much slower:

```bash
PYTENSOR_FLAGS="cxx=" python refit.py
```

The compiler-free alternative was not verified to completion.

### What a refit creates

`results/refitted/` is created on demand and is ignored by Git. A repeated
refit replaces its generated estimates, but never overwrites the published
reference tables or root-level saved models.

| Generated file or group | Purpose |
|---|---|
| `predictions.csv`, `scores.csv`, `coefficients.csv` | Fresh predictive outputs, aggregate performance and posterior coefficient summaries. |
| `diagnostics.csv` | Fresh sampling summaries, identified by experiment name. This differs from the reference filename `sampling_diagnostics.csv`. |
| `prior_summary.csv` | Fresh prior-predictive quantiles for the specified development fits. |
| `benchmark_comparison.csv`, `size_comparison.csv`, `development_comparisons.csv`, `selected_models.csv` | Fresh comparisons and cutoff-level choices within the documented experiment plan. |
| `size_comparisons.csv`, `size_performance.csv` | Fresh subgroup comparisons and predictive metrics. |
| `historical_forecasts.csv` | Fresh sequential forecasts for the same six reference example issuers. |
| `activity_distributions.csv`, `activity_models.csv`, `missing_outcomes.csv` | Recalculated descriptive trajectories, decay fits and missing-outcome bounds. |
| `models/<experiment name>.npz` | One saved fitted model for each row of `data/experiments.csv`. |
| `models/day1_baseline.npz` and `models/day<cutoff>_<model label>.npz` | Readable copies of the fresh baseline and selected cutoff models; labels can change if selection changes. |
| `models/prior_check_day10.npz` | Fresh prior-predictive draws for the representative day-10 fit. |
| `models/historical_forecasts.npz` | Fresh predictive draws for the six historical demonstrations. |

The script does **not** regenerate `benchmark_sensitivity.csv`,
`forecast_checks.csv`, the IPO-universe construction, the independent
initial-return matching or the SEC extraction pipeline.

A refit is not automatically connected to the notebook. The notebook continues
to use the reference `results/` and `models/` folders. Some fresh result
schemas and model filenames also differ, so simply changing one folder path
is not a supported replacement workflow. Integrating fresh results requires
a deliberate update of table loading, model loading and numerical statements
in the narrative.

## 4. Data folder: frozen inputs

The observation key is `ipo_id`. In the daily panel, `(ipo_id, day)` identifies
one event-time session. Monetary amounts are in US dollars. Returns used by the
models are log returns.

| File | Contents and role |
|---|---|
| `data/ipo_sample.csv` | 1,382 audited listings. Includes offering dates and proceeds, offer prices, sample-membership flags, exclusion reasons, providers, source-path references, adjustment information and outcome dates. `usable` selects the main modelling sample; `population` and `mature` define the population used for missing-outcome analysis. |
| `data/daily_prices.csv` | 82,215 event-panel rows and 10 columns: IPO identifier, session number, date, closing price, volume, broad and growth benchmark returns, VIX, daily stock log return and daily trading intensity. Contains the available histories of audited listings, not just usable IPOs. VIX is present but is not a fitted predictor. |
| `data/experiments.csv` | 86 documented predictive fits. Records each readable experiment name, phase, fold, training boundary, validation end, benchmark, model, cutoff, likelihood, size-control flag and prior-width multiplier. Sampler settings are defined in `fit_model()`, not in this table. |
| `data/initial_returns.csv` | 222 independently matched IPO observations, including IPOScoop offer and first-close values and an offer-match flag. Used for the descriptive initial-return validation, not as predictive-model inputs. |
| `data/offered_shares.csv` | 59 selective SEC offered-share validation observations. Includes extracted values, confidence and filing/source evidence, plus implied-share and error measures. Its 87 columns retain broader extraction provenance; most are not used by the predictive regressions. |

A source-path or local-path field is an audit reference. It does not imply that
the referenced raw file is included or that any script follows that path.
The source study documents Nasdaq/Yahoo histories, SEC records and IPOScoop
initial-return observations; the raw collection pipelines are not shipped.

## 5. Models folder: numeric draws

The five regression archives contain 3,200 posterior draws, predictor names,
training means and scales, cutoff/model metadata and reported diagnostics.
M0 controls for size; M1 adds early price performance; M2 adds trading intensity;
M3 additionally lets realised volatility affect predictive scale. At day 1,
early price performance and realised volatility are zero.

| File | Model or purpose |
|---|---|
| `models/day1_baseline.npz` | The final fitted size-conditioned M0 model for first-close-to-day-60 returns. Also supplies the pre-trading reference, reported as cutoff 0. This is a reference distribution, not an offer-price-to-day-60 price forecast. |
| `models/day1_intensity.npz` | The selected day-1 M2 model, using log trading intensity and log IPO proceeds. |
| `models/day5_volatility.npz` | The selected day-5 M3 model, using early excess return, log intensity, log proceeds and RV-dependent scale. |
| `models/day10_volatility.npz` | The selected day-10 M3 model, used for the coefficient-density plot, the live prefix check and the NEWCO example. |
| `models/day20_volatility.npz` | The selected day-20 M3 model for the shorter remaining horizon. |
| `models/prior_check_day10.npz` | A single `outcome` array with shape `(1, 500, 499)`: 500 prior-predictive draws for the 499 observations in the representative 2024 development training sample. Used for the prior-plausibility histogram. |
| `models/historical_forecasts.npz` | 30 arrays keyed by issuer identifier and cutoff: predictive draws for six historical IPOs at cutoffs 0, 1, 5, 10 and 20. Used for the historical density plots. |

These are compact replay artifacts, not complete sampler logs. In particular,
full sampling-statistic arrays needed to independently recheck all reported
divergence and BFMI diagnostics are not retained. The original 86 experiment
traces are not all included in the reference models folder.

## 6. Results folder: reference outputs

These files are the published snapshot, not temporary outputs from the last
refit. Scores are grouped by research phase, benchmark, likelihood, model and
cutoff where applicable.

| File | Contents and use |
|---|---|
| `results/predictions.csv` | 14,930 IPO-level prediction rows across the experiment phases: realised targets, event probabilities, predictive quantiles, log score, CRPS, PIT, coverage indicators and model/fold metadata. The notebook uses it for calibration; `check.py` checks the selected final models against it. |
| `results/scores.csv` | 53 aggregate score rows with sample counts, log score, CRPS, MAE, RMSE and 50/80/95% interval coverage with uncertainty bands. Supplies the final performance and calibration charts. |
| `results/coefficients.csv` | 121 posterior summaries, including coefficient medians, 95% credible intervals and probabilities of positive/negative signs. Used in the price and intensity model sections. |
| `results/benchmark_comparison.csv` | Four development scores comparing broad/growth benchmarks and Gaussian/Student-t errors. Displayed in the reference-model section. |
| `results/size_comparison.csv` | Four cutoff-specific development comparisons of a size-conditioned and unconditioned baseline. This singular filename is the size-control adoption check, not subgroup performance. |
| `results/development_comparisons.csv` | Twelve paired M1–M0, M2–M1 and M3–M2 comparisons, with log-score bootstrap intervals and CRPS differences. Supplies the incremental-signal tables. |
| `results/selected_models.csv` | Four frozen cutoff choices: M2 at day 1 and M3 at days 5, 10 and 20. Displayed and used for the reference calibration chart. |
| `results/prior_summary.csv` | Fifteen prior-predictive summaries for the specified 2024 development fits, reporting the median and 0.5/99.5% quantiles. Displayed alongside the prior-check chart. |
| `results/sampling_diagnostics.csv` | Eighty-six archived fit summaries with training counts, outcome boundaries, R-hat, ESS, divergence counts, BFMI and a diagnostic-pass flag. Available for inspection; not a full sampler trace. |
| `results/historical_forecasts.csv` | Thirty sequential forecast summaries for six historical IPOs, with issuer metadata, realised targets, medians, intervals and event probabilities. Used for the sequential demonstration and as the refit's example-issuer list. |
| `results/forecast_checks.csv` | One archived day-7 API-check result. It is displayed as an archived result, not executed anew; the current code separately executes day-10 and other validated-cutoff invariance checks. |
| `results/activity_distributions.csv` | 720 event-day quantile rows: six activity variables, two size groups and 60 sessions. Supplies the four cumulative/normalised activity charts. |
| `results/activity_models.csv` | Six constant/power-law/exponential decay fits across daily intensity and absolute returns, with slope uncertainty, held-out MSE and descriptive form comparisons. Displayed in the activity section. |
| `results/missing_outcomes.csv` | Sixty-four event-probability summaries across size thresholds, cutoffs and observability definitions. Reports identification bounds separately from sampling uncertainty. Supplies the missing-outcome table and chart. |
| `results/size_performance.csv` | 120 model-performance rows for size thresholds 0, $500m, $1bn and $2bn. Contains the large-IPO performance numbers discussed in Section 16; that section currently does not print this table. |
| `results/size_comparisons.csv` | Seventy-two paired model comparisons across phase, size threshold and cutoff. The notebook prints the final all-size subset earlier; Section 16 discusses large-IPO subsets stored here without printing them in that section. |
| `results/benchmark_sensitivity.csv` | Six final-cohort broad/growth benchmark comparison summaries. Retained for inspection, but not displayed by the current notebook or regenerated by `refit.py`. |

## 7. Reproducibility: what the evidence does and does not show

**Saved-snapshot replay works.** The notebook needs only this folder once
dependencies are installed. The audit reran a temporary copy in a fresh kernel:
19 code cells completed, all 18 figures regenerated and all 20 displayed tables
matched the saved table output. `check.py` passed for 976 selected-model
final-cohort forecasts and 30 historical updates. Aggregate score recomputation
also agreed with the reference to floating-point precision.

**Predictive refitting has been tested.** In the preceding build verification,
all 86 documented predictive fits completed with acceptable reported
diagnostics, and their 53 aggregate log-score and CRPS results matched the
reference values in that environment. The current audit reran replay and
numerical checks; it did not repeat all 86 fits. Some bootstrap summaries can
differ from the archived values even when point estimates agree.

**This is not complete raw-to-report reproduction.** Missing raw-download
caches, universe-construction inputs and SEC extraction code prevent independent
reconstruction of the prepared inputs. The day-7 check and additional benchmark
sensitivity table remain archived. A refit does not automatically refresh the
notebook, and the saved metadata do not constitute a complete sampler audit.

Direct package versions are pinned, but transitive dependencies, compiler
versions and operating-system details are not fully locked. Exact agreement
in one tested environment is not a guarantee of bit-for-bit agreement on every
machine. Git is initialised locally; the current project has no committed
snapshot or experiment history yet.

## 8. Research interpretation and limitations

The target is:

```text
remaining excess log return
= log(Close_60 / Close_cutoff)
  - sum(benchmark log returns for sessions cutoff+1 through 60)
```

Trading intensity is the cumulative proxy `Close × Volume / IPO proceeds`,
not trade-by-trade dollar turnover, free-float turnover or signed order flow.
Offer price is checked as listing metadata but is not a separate predictor of
the retained models.

Keep the following limits in mind:

- Final-cohort predictive intervals under-cover. For M3 at day 10, nominal 80%
  and 95% intervals cover about 67.6% and 86.9% of outcomes.
- The final $1bn+ and $2bn+ samples contain 18 and 4 IPOs. Subgroup improvements
  are exploratory, not strong evidence for a separate mega-IPO model.
- The paired score bootstrap resamples IPOs independently. Common calendar
  shocks and overlapping return windows may make its uncertainty too narrow.
- The usable sample requires complete future histories. Prefix-invariance
  checks verify feature construction; they do not remove this sample-selection
  condition or establish point-in-time vendor correctness.
- Activity ratios use each IPO's future sessions 41–60. They are retrospective
  descriptions, not live predictors. The activity-decay training cohort is
  split by listing date, and 36 usable pre-2025 listings have outcomes crossing
  the 2025 boundary. Its best descriptive form is also chosen using the later
  cohort, so this exercise is not an untouched predictive validation.
- Priors are empirical-Bayes and training-data calibrated. Their calibration
  scale is held fixed during sampling.
- Separate cutoff models predict different remaining horizons; they are not
  one recursively updated Bayesian filter.
- Student-t log-return forecasts support probabilities and quantiles, but do
  not imply a finite arithmetic expected wealth after exponentiation.
- No transaction-cost-aware strategy, order-book mechanism or identification
  of passive-fund demand is estimated.

The notebook retains the original markdown, including wording and presentation
issues. In particular, Section 13 describes an archived day-7 check and Section
16 refers to tables not printed there. The code comments and catalogue above
distinguish those claims from checks actually executed in this repository.

## 9. Useful technical references

- [NumPy array archives](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html):
  the storage format for numeric posterior and predictive draws.
- [NBClient notebook execution](https://nbclient.readthedocs.io/en/latest/client.html):
  fresh-kernel execution and explicit kernel selection.
- [scikit-learn BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html):
  a possible simpler baseline for future work, not a replacement preserving
  this study's Student-t likelihood or RV-dependent predictive scale.
