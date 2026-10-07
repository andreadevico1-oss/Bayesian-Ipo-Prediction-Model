"""Fit the small Bayesian regressions and calculate predictive distributions."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from features import early_features

SEED = 20260908


def predictor_names(kind, cutoff, size_conditioned=True):
    names = []
    if kind not in ["M0", "M0size"] and cutoff > 1:
        names.append("early_car")
    if kind in ["M2", "M3"]:
        names.append("log_ti")
    if size_conditioned or kind in ["M0size", "M1size"]:
        names.append("log_size")
    if kind == "M1size":
        names.append("price_size")
    return names


def standardize(data, names, means, scales):
    base = [name for name in names if name != "price_size"]
    values = data[base].to_numpy(dtype=float)
    values = (values - means) / scales
    if "price_size" in names:
        interaction = values[:, base.index("early_car")] * values[:, base.index("log_size")]
        values = np.column_stack([values, interaction])
    return values


def load_model(path):
    """Each file holds numeric arrays and a few scalar settings; no object loading."""
    with np.load(path, allow_pickle=False) as archive:
        model = {name: archive[name] for name in archive.files}
    for name, value in model.items():
        if value.ndim == 0:
            model[name] = value.item()
    model["names"] = model["names"].tolist()
    return model


def fit_model(train, kind, cutoff, likelihood="student", size_conditioned=True,
              prior_multiplier=1.0, draws=800, tune=800):
    """PyMC handles sampling; scaling and priors use only this training sample."""
    import arviz as az
    import pymc as pm

    names = predictor_names(kind, cutoff, size_conditioned)
    base = [name for name in names if name != "price_size"]
    means = train[base].to_numpy(dtype=float).mean(axis=0)
    scales = train[base].to_numpy(dtype=float).std(axis=0)
    scales = np.where(scales > 1e-10, scales, 1.0)
    x = standardize(train, names, means, scales)
    rv_mean = float(train.rv.mean())
    rv_sd = float(train.rv.std(ddof=0)) or 1.0
    rv = (train.rv.to_numpy() - rv_mean) / rv_sd
    y = train.y.to_numpy(dtype=float)
    outcome_scale = max((np.quantile(y, 0.75) - np.quantile(y, 0.25)) / 1.349,
                        np.std(y) * 0.1, 1e-4)

    with pm.Model(coords={"coefficient": names} if names else None):
        alpha = pm.Normal("alpha", 0, outcome_scale * prior_multiplier)
        location = alpha
        if names:
            beta = pm.Normal("beta", 0, outcome_scale * 0.5 * prior_multiplier,
                             dims="coefficient")
            location = alpha + pm.math.dot(x, beta)
        log_scale = pm.Normal("log_scale", np.log(outcome_scale), 0.5 * prior_multiplier)
        scale = pm.math.exp(log_scale)
        if kind == "M3" and cutoff > 1:
            delta = pm.Normal("delta_rv", 0, 0.35 * prior_multiplier)
            scale = pm.math.exp(log_scale + delta * rv)
        if likelihood == "student":
            nu = pm.Deterministic("nu", 2 + pm.Exponential("nu_minus_two", 1 / 10))
            pm.StudentT("outcome", nu=nu, mu=location, sigma=scale, observed=y)
        else:
            pm.Normal("outcome", mu=location, sigma=scale, observed=y)
        prior = pm.sample_prior_predictive(samples=500, random_seed=SEED)
        trace = pm.sample(draws=draws, tune=tune, chains=4, cores=1, random_seed=SEED,
                          target_accept=0.9, progressbar=False,
                          idata_kwargs={"log_likelihood": False})

    parameters = [name for name in ["alpha", "beta", "log_scale", "delta_rv", "nu"]
                  if name in trace.posterior]
    summary = az.summary(trace, var_names=parameters)
    diagnostics = {
        "rhat_max": float(summary.r_hat.max()),
        "ess_bulk_min": float(summary.ess_bulk.min()),
        "ess_tail_min": float(summary.ess_tail.min()),
        "divergences": int(trace.sample_stats.diverging.sum()),
        "bfmi_min": float(az.bfmi(trace).min()),
    }
    if not (diagnostics["rhat_max"] <= 1.01 and diagnostics["ess_bulk_min"] >= 400
            and diagnostics["ess_tail_min"] >= 400 and diagnostics["divergences"] == 0
            and diagnostics["bfmi_min"] > 0.3):
        raise RuntimeError(f"Sampling diagnostics failed: {diagnostics}")

    model = {
        "kind": kind, "cutoff": cutoff, "likelihood": likelihood,
        "names": names, "means": means, "scales": scales,
        "rv_mean": rv_mean, "rv_sd": rv_sd,
        "n_train": len(train), "max_outcome_date": str(train.outcome_date.max()),
        "prior_outcomes": prior.prior_predictive.outcome.values,
        **diagnostics,
    }
    for name in parameters:
        values = trace.posterior[name].values
        model[name] = values.reshape((-1,) + values.shape[2:])
    if "beta" not in model:
        model["beta"] = np.empty((4 * draws, 0))
    if "nu" not in model:
        model["nu"] = np.full(4 * draws, np.inf)
    return model


def predictive_parameters(model, data):
    x = standardize(data, model["names"], model["means"], model["scales"])
    location = model["alpha"][:, None] + model["beta"] @ x.T
    scale = np.exp(model["log_scale"])[:, None] * np.ones((1, len(data)))
    if "delta_rv" in model:
        rv = (data.rv.to_numpy() - model["rv_mean"]) / model["rv_sd"]
        scale *= np.exp(model["delta_rv"][:, None] * rv[None, :])
    return location, scale, model["nu"][:, None]


def predictive_cdf(model, values, nu):
    if model["likelihood"] == "student":
        return stats.t.cdf(values, nu)
    return stats.norm.cdf(values)


def predict(model, data, seed=SEED):
    """Return IPO-level probabilities, intervals and proper scores."""
    location, scale, nu = predictive_parameters(model, data)
    rng = np.random.default_rng(seed)
    if model["likelihood"] == "student":
        draws = location + scale * rng.standard_t(nu, size=location.shape)
    else:
        draws = location + scale * rng.normal(size=location.shape)

    output = data[["ipo_id", "ipo_date", "ipo_proceeds"]].copy()
    output["p_positive"] = 1 - predictive_cdf(model, -location / scale, nu).mean(axis=0)
    output["p_loss20"] = predictive_cdf(model, (np.log(0.8) - location) / scale, nu).mean(axis=0)
    for probability in [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]:
        output[f"q{probability:g}"] = np.quantile(draws, probability, axis=0)
    if "y" in data:
        actual = data.y.to_numpy()
        standardized = (actual - location) / scale
        if model["likelihood"] == "student":
            log_density = stats.t.logpdf(standardized, nu) - np.log(scale)
        else:
            log_density = stats.norm.logpdf(standardized) - np.log(scale)
        output["y"] = actual
        output["log_score"] = logsumexp(log_density, axis=0) - np.log(len(draws))
        ordered = np.sort(draws, axis=0)
        count = len(draws)
        weights = (2 * np.arange(1, count + 1) - count - 1)[:, None]
        output["crps"] = np.abs(draws - actual).mean(axis=0) - (weights * ordered).sum(axis=0) / count ** 2
        output["pit"] = predictive_cdf(model, standardized, nu).mean(axis=0)
        for level, lower, upper in [(50, 0.25, 0.75), (80, 0.1, 0.9), (95, 0.025, 0.975)]:
            output[f"cover{level}"] = output.y.between(output[f"q{lower:g}"], output[f"q{upper:g}"])
    return output, draws


def forecast(model, *, offer_price, ipo_proceeds, ipo_date, prices=None,
             volume=None, benchmark_returns=None):
    """Forecast a new IPO with a reference model or an observed trading prefix."""
    if not np.isfinite([offer_price, ipo_proceeds]).all() or min(offer_price, ipo_proceeds) <= 0:
        raise ValueError("Positive offer price and IPO proceeds required.")
    listing_date = pd.Timestamp(ipo_date)
    if pd.isna(listing_date) or listing_date < pd.Timestamp("2025-01-01"):
        raise ValueError("The target must be listed on or after 2025-01-01.")
    if pd.Timestamp(model["max_outcome_date"]) >= listing_date:
        raise ValueError("Training outcomes were unavailable before the IPO.")
    cutoff = model["cutoff"]
    if prices is None:
        if model["kind"] != "M0":
            raise ValueError("Pre-trading forecasts require a baseline model.")
        values = {"log_size": np.log(ipo_proceeds), "rv": 0.0}
        cutoff = 0
    else:
        if volume is None or benchmark_returns is None:
            raise ValueError("Supply prices, volume and benchmark returns together.")
        # Slice before checking data: the future suffix is outside the information set.
        prices = np.asarray(prices)[:cutoff]
        volume = np.asarray(volume)[:cutoff]
        benchmark_returns = np.asarray(benchmark_returns)[:cutoff]
        if any(len(values) != cutoff for values in [prices, volume, benchmark_returns]):
            raise ValueError("Insufficient observed trading sessions.")
        tape = pd.DataFrame({"day": np.arange(1, cutoff + 1), "Close": prices,
                             "Volume": volume, "growth_return": benchmark_returns})
        values = early_features(ipo_proceeds, tape, cutoff)
    row = pd.DataFrame([{**values, "ipo_id": "new_ipo", "ipo_date": listing_date,
                         "ipo_proceeds": ipo_proceeds}])
    output, draws = predict(model, row)
    result = {
        "calibration_status": "Research forecast: broad-cohort intervals undercovered; $1bn+ validation has only 18 IPOs",
        "cutoff": cutoff, "model": model["kind"], "draws_log_excess_return": draws[:, 0],
        "p_positive": float(output.p_positive.iloc[0]),
        "p_loss20": float(output.p_loss20.iloc[0]), "median_log": float(output["q0.5"].iloc[0]),
    }
    for level, lower, upper in [(50, "q0.25", "q0.75"), (80, "q0.1", "q0.9"), (95, "q0.025", "q0.975")]:
        result[f"interval{level}_log"] = (float(output[lower].iloc[0]), float(output[upper].iloc[0]))
    return result
