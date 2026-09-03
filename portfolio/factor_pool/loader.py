"""Factor CSV loader. Only keeps date/instrument/factor columns."""
from __future__ import annotations

import pathlib
from typing import Mapping

import pandas as pd

REQUIRED_COLUMNS = ("date", "instrument", "factor")


def load_factor_csv(path: str | pathlib.Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.rename(columns={"factor": name})
    return df[["date", "instrument", name]]


def load_many(paths: Mapping[str, str | pathlib.Path]) -> dict[str, pd.DataFrame]:
    return {name: load_factor_csv(p, name=name) for name, p in paths.items()}
