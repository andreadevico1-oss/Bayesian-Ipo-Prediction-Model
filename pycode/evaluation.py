"""Forecast scores, IPO bootstrap comparisons and robustness calculations."""

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260908


def score_table(predictions):
    rows = []
    columns = ["phase", "benchmark", "likelihood", "kind", "cutoff"]
    for key, group in predictions.groupby(columns, sort=False):
        row = dict(zip(columns, key))
        errors = group.y - group["q0.5"]
        row.update(n=len(group), log_score=group.log_score.mean(), crps=group.crps.mean(),
                   mae=errors.abs().mean(), rmse=np.sqrt((errors * errors).mean()))
        for level in [50, 80, 95]:
            covered = int(group[f"cover{level}"].sum())
            lower, upper = stats.beta.ppf([0.025, 0.975], covered + 0.5,
                                          len(group) - covered + 0.5)
            row[f"coverage{level}"] = covered / len(group)
            row[f"coverage{level}_lo"] = lower
            row[f"coverage{level}_hi"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_gain(richer, simpler, repetitions=5000):
    """Paired resampling keeps both forecasts for the same IPO together."""
    pair = richer.merge(simpler, on="ipo_id", suffixes=("_richer", "_simpler"),
                        validate="one_to_one")
    if pair.empty:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "crps_delta": np.nan}
    gains = (pair.log_score_richer - pair.log_score_simpler).to_numpy()
    rng = np.random.default_rng(SEED)
    averages = np.empty(repetitions)
    for repetition in range(repetitions):
        averages[repetition] = rng.choice(gains, size=len(gains), replace=True).mean()
    lower, upper = np.quantile(averages, [0.025, 0.975])
    return {"delta": gains.mean(), "lo": lower, "hi": upper, "n": len(pair),
            "crps_delta": (pair.crps_richer - pair.crps_simpler).mean()}


def model_comparisons(predictions, thresholds=(0, 500e6, 1e9, 2e9)):
    rows = []
    for phase in ["development", "final"]:
        for threshold in thresholds:
            cutoffs = [5, 10, 20]
            if threshold == 0 and phase == "development":
                cutoffs = [1, 5, 10, 20]
            for cutoff in cutoffs:
                subset = predictions[(predictions.phase == phase)
                                     & (predictions.ipo_proceeds >= threshold)
                                     & (predictions.cutoff == cutoff)]
                for richer, simpler in [("M1", "M0"), ("M2", "M1"), ("M3", "M2")]:
                    gain = bootstrap_gain(subset[subset.kind == richer], subset[subset.kind == simpler])
                    rows.append({"phase": phase, "threshold": threshold, "cutoff": cutoff,
                                 "comparison": f"{richer}-{simpler}", **gain})
    return pd.DataFrame(rows)


def size_performance(predictions):
    rows = []
    for phase in ["development", "final"]:
        for threshold in [0, 500e6, 1e9, 2e9]:
            subset = predictions[(predictions.phase == phase) & (predictions.ipo_proceeds >= threshold)]
            subset = subset[subset.kind.isin(["M0", "M1", "M2", "M3"])]
            summary = score_table(subset)
            for row in summary.to_dict("records"):
                group = subset[(subset.kind == row["kind"]) & (subset.cutoff == row["cutoff"])]
                row.update(threshold=threshold, p_positive=group.p_positive.mean(),
                           observed_positive=(group.y > 0).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def missing_outcome_bounds(audit, panel):
    """Identification bounds and Jeffreys sampling intervals from available outcomes."""
    from features import remaining_return

    population = audit[audit.population & audit.mature].copy()
    histories = dict(tuple(panel.groupby("ipo_id")))
    outcomes = {}
    for cutoff in [1, 5, 10, 20]:
        outcomes[cutoff] = population.ipo_id.map(
            lambda issuer_id: remaining_return(histories[issuer_id], cutoff)
            if issuer_id in histories else np.nan
        )
    rows = []
    for threshold in [0, 500e6, 1e9, 2e9]:
        eligible = population if threshold == 0 else population[population.ipo_proceeds >= threshold]
        for cutoff in [1, 5, 10, 20]:
            for scope in ["endpoint", "joint_tape"]:
                values = outcomes[cutoff].loc[eligible.index]
                observed = values.notna()
                if scope == "joint_tape":
                    observed &= eligible.usable
                values = values[observed]
                total, count = len(eligible), len(values)
                fraction = count / total
                for event in ["positive", "loss20"]:
                    successes = int((values > 0).sum() if event == "positive"
                                    else (values < np.log(0.8)).sum())
                    lower, median, upper = stats.beta.ppf(
                        [0.025, 0.5, 0.975], successes + 0.5, count - successes + 0.5
                    )
                    rows.append({
                        "threshold": threshold, "cutoff": cutoff, "scope": scope, "event": event,
                        "N": total, "n": count, "missing": total - count, "pi": fraction,
                        "width": 1 - fraction, "observed_rate": successes / count,
                        "q_median": median, "q_lo": lower, "q_hi": upper,
                        "lower_median": fraction * median,
                        "upper_median": fraction * median + 1 - fraction,
                        "lower_lo": fraction * lower, "lower_hi": fraction * upper,
                        "upper_lo": fraction * lower + 1 - fraction,
                        "upper_hi": fraction * upper + 1 - fraction,
                    })
    return pd.DataFrame(rows)


def activity_trajectories(audit, panel):
    """Normalise daily activity by each issuer's median in sessions 41–60."""
    usable = audit[audit.usable][["ipo_id", "ipo_date", "ipo_proceeds"]]
    data = panel.merge(usable, on="ipo_id", how="inner", validate="many_to_one")
    data = data.sort_values(["ipo_id", "day"])
    data["abs_return"] = data["return"].abs()
    data["cum_ti"] = data.groupby("ipo_id").daily_ti.cumsum()
    squared = data["return"].fillna(0) ** 2
    data["cum_rv"] = np.sqrt(squared.groupby(data.ipo_id).cumsum())
    for column in ["daily_ti", "abs_return"]:
        baseline = data[data.day >= 41].groupby("ipo_id")[column].median()
        data[column + "_ratio"] = data[column] / data.ipo_id.map(baseline.replace(0, np.nan))
    return data


def activity_distributions(data):
    rows = []
    for group in ["all", "$1bn+"]:
        subset = data if group == "all" else data[data.ipo_proceeds >= 1e9]
        for variable in ["daily_ti", "cum_ti", "cum_rv", "daily_ti_ratio", "abs_return_ratio", "abs_return"]:
            quantiles = subset.groupby("day")[variable].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
            quantiles.columns = ["q10", "q25", "q50", "q75", "q90"]
            quantiles = quantiles.reset_index()
            quantiles["variable"] = variable
            quantiles["group"] = group
            rows.append(quantiles)
    return pd.concat(rows, ignore_index=True)


def fit_activity_decay(data, repetitions=2000):
    """OLS descriptions with bootstrap uncertainty across whole IPO trajectories."""
    rows = []
    for variable in ["daily_ti", "abs_return"]:
        subset = data[data[variable + "_ratio"] > 0].copy()
        subset["log_activity"] = np.log(subset[variable + "_ratio"])
        train = subset[subset.ipo_date < "2025-01-01"]
        test = subset[subset.ipo_date >= "2025-01-01"]
        losses_by_form = {}
        for form in ["constant", "power", "exponential"]:
            def design(frame):
                if form == "constant":
                    return np.ones((len(frame), 1))
                day = np.log(frame.day) if form == "power" else frame.day
                return np.column_stack([np.ones(len(frame)), day])

            x, y = design(train), train.log_activity.to_numpy()
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            test_x = design(test)
            losses = (test.log_activity.to_numpy() - test_x @ beta) ** 2
            # Give each held-out IPO equal weight, even if a zero ratio was dropped.
            issuer_losses = pd.Series(losses, index=test.ipo_id).groupby(level=0).mean()
            losses_by_form[form] = issuer_losses
            slope_lower = slope_upper = np.nan
            if form != "constant":
                # Precompute sums by IPO; resampling them resamples full trajectories.
                blocks = []
                for issuer_id, positions in train.groupby("ipo_id").indices.items():
                    blocks.append((x[positions].T @ x[positions], x[positions].T @ y[positions]))
                rng = np.random.default_rng(SEED)
                slopes = []
                for _ in range(repetitions):
                    sample = rng.integers(0, len(blocks), len(blocks))
                    xtx = sum(blocks[index][0] for index in sample)
                    xty = sum(blocks[index][1] for index in sample)
                    slopes.append(np.linalg.solve(xtx, xty)[1])
                slope_lower, slope_upper = np.quantile(slopes, [0.025, 0.975])
            rows.append({
                "variable": variable, "form": form,
                "n_train": train.ipo_id.nunique(), "n_test": test.ipo_id.nunique(),
                "intercept": beta[0], "slope": beta[1] if len(beta) > 1 else np.nan,
                "slope_lo": slope_lower, "slope_hi": slope_upper,
                "heldout_mse": issuer_losses.mean(), "half_life_days": np.nan,
            })
        best = min(losses_by_form, key=lambda form: losses_by_form[form].mean())
        alternatives = [form for form in losses_by_form if form != best]
        alternative = min(alternatives, key=lambda form: losses_by_form[form].mean())
        difference = (losses_by_form[best] - losses_by_form[alternative]).to_numpy()
        rng = np.random.default_rng(SEED)
        bootstrap = [rng.choice(difference, len(difference), replace=True).mean()
                     for _ in range(repetitions)]
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        for row in rows[-3:]:
            row.update(selected_form=best, selected_minus_alternative_mse=difference.mean(),
                       delta_lo=lower, delta_hi=upper)
    output = pd.DataFrame(rows)
    return output
