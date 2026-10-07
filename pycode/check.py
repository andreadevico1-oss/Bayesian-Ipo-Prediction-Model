"""Check chronology, information cutoffs and saved forecast agreement."""

from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_models import forecast, load_model, predict
from features import chronological_split, early_features, make_cutoff_data


def main():
    root = Path(__file__).resolve().parent
    audit = pd.read_csv(root / "data/ipo_sample.csv", parse_dates=["ipo_date", "outcome_date"])
    panel = pd.read_csv(root / "data/daily_prices.csv", parse_dates=["date"], float_precision="round_trip")
    expected = pd.read_csv(root / "results/historical_forecasts.csv", float_precision="round_trip")
    saved_predictions = pd.read_csv(root / "results/predictions.csv", float_precision="round_trip")
    assert audit.ipo_id.is_unique
    assert not panel.duplicated(["ipo_id", "day"]).any()
    assert audit.usable.sum() == 903

    baseline = load_model(root / "models/day1_baseline.npz")
    models = {1: load_model(root / "models/day1_intensity.npz")}
    for cutoff in [5, 10, 20]:
        models[cutoff] = load_model(root / "models" / f"day{cutoff}_volatility.npz")

    for cutoff in [1, 5, 10, 20]:
        data = make_cutoff_data(audit, panel, cutoff)
        for start, end, train_count, test_count in [
            ("2022-01-01", "2024-01-01", 304, 159),
            ("2024-01-01", "2025-01-01", 499, 139),
            ("2025-01-01", "2026-09-08", 623, 244),
        ]:
            train, test = chronological_split(data, start, end)
            assert len(train) == train_count and len(test) == test_count
            assert train.outcome_date.max() < pd.Timestamp(start)
            assert train.feature_date.max() < pd.Timestamp(start)
            assert set(train.ipo_id).isdisjoint(test.ipo_id)
        actual, _ = predict(models[cutoff], test)
        reference = saved_predictions[(saved_predictions.phase == "final")
                                      & (saved_predictions.cutoff == cutoff)
                                      & (saved_predictions.kind == models[cutoff]["kind"])]
        pair = actual.merge(reference, on="ipo_id", suffixes=("_new", "_saved"),
                            validate="one_to_one")
        assert len(pair) == 244
        for column in ["y", "p_positive", "p_loss20", "q0.5", "log_score", "crps", "pit"]:
            np.testing.assert_allclose(pair[column + "_new"], pair[column + "_saved"],
                                       rtol=1e-10, atol=1e-10)

    for row in expected.itertuples(index=False):
        issuer = audit.set_index("ipo_id").loc[row.ipo_id]
        history = panel[panel.ipo_id == row.ipo_id].sort_values("day")
        offer = {"offer_price": issuer.offer_price, "ipo_proceeds": issuer.ipo_proceeds,
                 "ipo_date": issuer.ipo_date}
        if row.cutoff == 0:
            result = forecast(baseline, **offer)
        else:
            observed = {"prices": history.Close.iloc[:row.cutoff],
                        "volume": history.Volume.iloc[:row.cutoff],
                        "benchmark_returns": history.growth_return.iloc[:row.cutoff]}
            result = forecast(models[row.cutoff], **offer, **observed)
            # Deliberately invalid future values must not change a prefix forecast.
            poisoned = {name: np.r_[values, np.nan, -100] for name, values in observed.items()}
            future_result = forecast(models[row.cutoff], **offer, **poisoned)
            np.testing.assert_array_equal(result["draws_log_excess_return"],
                                          future_result["draws_log_excess_return"])
            prefix_features = early_features(issuer.ipo_proceeds, history, row.cutoff)
            changed = history.copy()
            changed.loc[changed.day > row.cutoff, ["Close", "Volume", "growth_return"]] = np.nan
            assert prefix_features == early_features(issuer.ipo_proceeds, changed, row.cutoff)
        for field in ["median_log", "p_positive", "p_loss20"]:
            np.testing.assert_allclose(result[field], getattr(row, field), rtol=1e-12, atol=1e-12)
        for level in [50, 80, 95]:
            np.testing.assert_allclose(result[f"interval{level}_log"],
                                       [getattr(row, f"lo{level}"), getattr(row, f"hi{level}")],
                                       rtol=1e-12, atol=1e-12)

    try:
        forecast(baseline, offer_price=10, ipo_proceeds=1e9, ipo_date="2024-01-01")
    except ValueError:
        pass
    else:
        raise AssertionError("A pre-training IPO should have been rejected.")
    print("Passed: chronological splits, 976 held-out forecasts and 30 historical updates.")
    print("Passed: future-suffix invariance and the frozen training boundary.")


if __name__ == "__main__":
    main()
