"""Price, volume and return calculations used by the notebook."""

import numpy as np
import pandas as pd


def early_features(ipo_proceeds, tape, cutoff, benchmark="growth"):
    """Calculate signals using sessions 1 through the cutoff only."""
    if not isinstance(cutoff, (int, np.integer)) or not 1 <= cutoff < 60:
        raise ValueError("The cutoff must be an integer between 1 and 59.")
    prefix = tape.sort_values("day")
    prefix = prefix[prefix.day <= cutoff]
    if prefix.day.tolist() != list(range(1, cutoff + 1)):
        raise ValueError("Missing or duplicate trading session.")
    if not np.isfinite(ipo_proceeds) or ipo_proceeds <= 0:
        raise ValueError("IPO proceeds must be positive.")

    prices = prefix.Close.to_numpy(dtype=float)
    volume = prefix.Volume.to_numpy(dtype=float)
    market = prefix[f"{benchmark}_return"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and positive.")
    if not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError("Volume must be finite and nonnegative.")
    if not np.isfinite(market[1:]).all():
        raise ValueError("Missing benchmark return.")

    returns = np.diff(np.log(prices))
    intensity = np.sum(prices * volume) / ipo_proceeds
    return {
        "early_car": float(np.sum(returns - market[1:])),
        "ti": float(intensity),
        "log_ti": float(np.log1p(intensity)),
        "rv": float(np.sqrt(np.sum(returns * returns))),
        "log_size": float(np.log(ipo_proceeds)),
        "drawdown": float(np.min(prices / np.maximum.accumulate(prices) - 1)),
        "feature_date": pd.Timestamp(prefix.date.iloc[-1]) if "date" in prefix else pd.NaT,
    }


def remaining_return(tape, cutoff, benchmark="growth"):
    """Calculate the realised excess log return from the cutoff close to day 60."""
    history = tape.sort_values("day").set_index("day").reindex(range(1, 61))
    prices = history.Close.to_numpy(dtype=float)
    market = history[f"{benchmark}_return"].to_numpy(dtype=float)
    if not np.isfinite(prices[cutoff - 1:]).all():
        return np.nan
    if not np.isfinite(market[cutoff:]).all():
        return np.nan
    return float(np.log(prices[-1] / prices[cutoff - 1]) - market[cutoff:].sum())


def make_cutoff_data(audit, panel, cutoff, benchmark="growth"):
    """Build one row per usable IPO."""
    histories = dict(tuple(panel.groupby("ipo_id")))
    rows = []
    for _, issuer in audit[audit.usable].iterrows():
        history = histories[issuer.ipo_id]
        row = issuer.to_dict()
        row.update(early_features(issuer.ipo_proceeds, history, cutoff, benchmark))
        row["y"] = remaining_return(history, cutoff, benchmark)
        row["cutoff"] = cutoff
        row["benchmark"] = benchmark
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ipo_date", "ipo_id"]).reset_index(drop=True)


def chronological_split(data, start, end):
    """Training outcomes must be available before the validation period starts."""
    train = data[(data.ipo_date < start) & (data.outcome_date < start)]
    test = data[(data.ipo_date >= start) & (data.ipo_date < end)]
    return train, test
