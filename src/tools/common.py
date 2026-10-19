
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(path, required_columns=None):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}"
        )

    df = pd.read_csv(path)

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {path}: {sorted(missing)}"
            )

    return df


def normalize_date(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def select_snapshot(df, portfolio_id, as_of_date):
    required = {"portfolio_id", "date"}

    if not required.issubset(df.columns):
        raise ValueError("Missing portfolio_id or date")

    dates = pd.to_datetime(df["date"], errors="raise")

    selected = df.loc[
        (df["portfolio_id"] == portfolio_id)
        & (dates == pd.Timestamp(as_of_date))
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one observation for "
            f"{portfolio_id} on {as_of_date}; "
            f"found {len(selected)}"
        )

    return selected.iloc[0]


def json_safe(value):
    if value is None:
        return None

    if isinstance(value, (pd.Timestamp,)):
        return normalize_date(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if pd.isna(value):
        return None

    return value
