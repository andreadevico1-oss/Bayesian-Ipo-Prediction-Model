"""Refit the documented experiments from the frozen daily data."""

from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_models import fit_model, forecast, predict
from evaluation import (activity_distributions, activity_trajectories, bootstrap_gain,
                        fit_activity_decay, missing_outcome_bounds, model_comparisons,
                        score_table, size_performance)
from features import chronological_split, make_cutoff_data, remaining_return


def main():
    root = Path(__file__).resolve().parent.parent
    data_folder = root / "data"
    output = root / "results" / "refitted"
    model_folder = output / "models"
    model_folder.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(data_folder / "ipo_sample.csv", parse_dates=["ipo_date", "outcome_date"])
    panel = pd.read_csv(data_folder / "daily_prices.csv", parse_dates=["date"], float_precision="round_trip")
    experiments = pd.read_csv(data_folder / "experiments.csv", dtype={"fold": str})
    features = {}
    for benchmark in ["broad", "growth"]:
        for cutoff in [1, 5, 10, 20]:
            features[(benchmark, cutoff)] = make_cutoff_data(audit, panel, cutoff, benchmark)

    predictions, coefficients, diagnostics, prior_rows = [], [], [], []
    selected_models = {}
    for experiment in experiments.itertuples(index=False):
        data = features[(experiment.benchmark, experiment.cutoff)]
        train, test = chronological_split(data, experiment.start, experiment.end)
        print(experiment.name, f"train={len(train)}, test={len(test)}", flush=True)
        model = fit_model(train, experiment.kind, experiment.cutoff,
                          experiment.likelihood, experiment.size_conditioned,
                          experiment.prior_multiplier)
        np.savez_compressed(model_folder / f"{experiment.name}.npz", **model)
        result, draws = predict(model, test)
        for name in ["phase", "fold", "benchmark", "likelihood", "kind", "cutoff", "size_conditioned"]:
            result[name] = getattr(experiment, name)
        predictions.append(result)
        diagnostics.append({"name": experiment.name,
                            **{name: model[name] for name in ["n_train", "max_outcome_date", "rhat_max",
                                                             "ess_bulk_min", "ess_tail_min", "divergences", "bfmi_min"]}})
        parameter_draws = {name: model["beta"][:, index] for index, name in enumerate(model["names"])}
        if "delta_rv" in model:
            parameter_draws["delta_rv"] = model["delta_rv"]
        for name, values in parameter_draws.items():
            lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
            coefficients.append({"coefficient": name, "median": median, "lo": lower, "hi": upper,
                                 "p_positive": (values > 0).mean(), "p_negative": (values < 0).mean(),
                                 "size_conditioned": experiment.size_conditioned, "kind": experiment.kind,
                                 "cutoff": experiment.cutoff, "benchmark": experiment.benchmark,
                                 "phase": experiment.phase, "fold": experiment.fold,
                                 "prior_multiplier": experiment.prior_multiplier})
        if experiment.phase == "development" and experiment.fold == "2024":
            quantiles = np.quantile(model["prior_outcomes"], [0.005, 0.5, 0.995])
            prior_rows.append({"kind": experiment.kind, "cutoff": experiment.cutoff,
                               "prior_0.5%": quantiles[0], "prior_median": quantiles[1], "prior_99.5%": quantiles[2]})
            if experiment.kind == "M0" and experiment.cutoff == 10:
                np.savez_compressed(model_folder / "prior_check_day10.npz", outcome=model["prior_outcomes"])
        if experiment.phase == "final" and experiment.benchmark == "growth":
            selected_models[(experiment.cutoff, experiment.kind)] = model

    all_predictions = pd.concat(predictions, ignore_index=True)
    # On day 1, early CAR and RV are zero: M1=M0 and M3=M2.
    aliases = []
    for source, alias in [("M0", "M1"), ("M2", "M3")]:
        equivalent = all_predictions[(all_predictions.phase == "development")
                                     & (all_predictions.cutoff == 1)
                                     & (all_predictions.kind == source)].copy()
        equivalent["kind"] = alias
        aliases.append(equivalent)
    all_predictions = pd.concat([all_predictions, *aliases], ignore_index=True)
    all_predictions.to_csv(output / "predictions.csv", index=False)
    score_table(all_predictions).to_csv(output / "scores.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output / "coefficients.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "diagnostics.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(output / "prior_summary.csv", index=False)
    screen = all_predictions[(all_predictions.phase == "reference_screen") & (all_predictions.kind == "M0")]
    screen.groupby(["benchmark", "likelihood"]).log_score.mean().reset_index().to_csv(
        output / "benchmark_comparison.csv", index=False
    )
    main_predictions = all_predictions[(all_predictions.benchmark == "growth")
                                       & (all_predictions.likelihood == "student")]
    comparisons = model_comparisons(main_predictions)
    comparisons.to_csv(output / "size_comparisons.csv", index=False)
    size_performance(main_predictions).to_csv(output / "size_performance.csv", index=False)
    development = comparisons[(comparisons.phase == "development") & (comparisons.threshold == 0)].copy()
    development["comparison"] = development.cutoff.astype(str) + ":" + development.comparison
    development[["comparison", "delta", "lo", "hi", "n", "crps_delta"]].to_csv(
        output / "development_comparisons.csv", index=False
    )

    gates, selected = [], []
    for cutoff in [1, 5, 10, 20]:
        screen = main_predictions[(main_predictions.phase == "reference_screen") & (main_predictions.cutoff == cutoff)]
        gate = bootstrap_gain(screen[screen.kind == "M0size"], screen[screen.kind == "M0"])
        gates.append({"cutoff": cutoff, **gate})
        subset = main_predictions[(main_predictions.phase == "development") & (main_predictions.cutoff == cutoff)]
        choice = "M0"
        for richer, simpler in [("M1", "M0"), ("M2", "M1"), ("M3", "M2")]:
            gain = bootstrap_gain(subset[subset.kind == richer], subset[subset.kind == simpler])
            if gain["lo"] > 0 and gain["crps_delta"] <= 0:
                choice = richer
        if cutoff == 1:
            choice = "M2" if choice in ["M2", "M3"] else "M0"
        selected.append({"cutoff": cutoff, "model": choice})
    pd.DataFrame(gates).to_csv(output / "size_comparison.csv", index=False)
    pd.DataFrame(selected).to_csv(output / "selected_models.csv", index=False)

    # Preserve the example issuer selection, not their published forecast values.
    issuer_ids = pd.read_csv(root / "results/historical_forecasts.csv").ipo_id.unique()
    chosen = {row["cutoff"]: row["model"] for row in selected}
    baseline = selected_models[(1, "M0")]
    np.savez_compressed(model_folder / "day1_baseline.npz", **baseline)
    for cutoff, kind in chosen.items():
        label = {"M0": "baseline", "M1": "price", "M2": "intensity", "M3": "volatility"}[kind]
        np.savez_compressed(model_folder / f"day{cutoff}_{label}.npz",
                            **selected_models[(cutoff, kind)])
    history_rows, history_draws = [], {}
    for issuer_id in issuer_ids:
        issuer = audit.set_index("ipo_id").loc[issuer_id]
        history = panel[panel.ipo_id == issuer_id].sort_values("day")
        offer = {"offer_price": issuer.offer_price, "ipo_proceeds": issuer.ipo_proceeds,
                 "ipo_date": issuer.ipo_date}
        for cutoff in [0, 1, 5, 10, 20]:
            if cutoff == 0:
                result = forecast(baseline, **offer)
            else:
                result = forecast(selected_models[(cutoff, chosen[cutoff])], **offer,
                                  prices=history.Close, volume=history.Volume,
                                  benchmark_returns=history.growth_return)
            row = {"ipo_id": issuer_id, "ticker": issuer.ticker,
                   "ipo_date": issuer.ipo_date, "ipo_proceeds": issuer.ipo_proceeds,
                   "cutoff": cutoff, "model": result["model"],
                   "median_log": result["median_log"], "p_positive": result["p_positive"],
                   "p_loss20": result["p_loss20"],
                   "realized": remaining_return(history, max(cutoff, 1))}
            for level in [50, 80, 95]:
                row[f"lo{level}"], row[f"hi{level}"] = result[f"interval{level}_log"]
            history_rows.append(row)
            history_draws[f"{issuer_id}_{cutoff}"] = result["draws_log_excess_return"]
    pd.DataFrame(history_rows).to_csv(output / "historical_forecasts.csv", index=False)
    np.savez_compressed(model_folder / "historical_forecasts.npz", **history_draws)

    activity = activity_trajectories(audit, panel)
    activity_distributions(activity).to_csv(output / "activity_distributions.csv", index=False)
    fit_activity_decay(activity).to_csv(output / "activity_models.csv", index=False)
    missing_outcome_bounds(audit, panel).to_csv(output / "missing_outcomes.csv", index=False)
    print("Refitting complete. Fresh estimates are in results/refitted/.")
    print("The published notebook and its reference estimates have not been overwritten.")


if __name__ == "__main__":
    main()
