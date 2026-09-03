"""Single-writer, staged HDF5 persistence for crypto-quant tables."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

import pandas as pd

from .schemas import TABLE_SPECS, normalize_table


class StoreLockedError(RuntimeError):
    pass


class StagingMismatchError(RuntimeError):
    pass


def _metadata_frame(updates: Mapping[str, object]) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return pd.DataFrame({"key": list(updates), "value": [json.dumps(v, sort_keys=True) for v in updates.values()], "updated_at_utc": [now] * len(updates)})


@contextmanager
def single_writer_lock(lock_path: Path) -> Iterator[None]:
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StoreLockedError(f"store lock is held: {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class CryptoQuantStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, name: str, frame: pd.DataFrame, *, append: bool = False) -> None:
        spec = TABLE_SPECS[name]
        normalized = normalize_table(name, frame)
        with pd.HDFStore(self.path, mode="a") as hdf:
            hdf.append(name, normalized, format="table", data_columns=list(spec.data_columns), min_itemsize=spec.min_itemsize, append=append, index=False)

    def read(self, name: str, where: str | None = None) -> pd.DataFrame:
        if name == "_metadata":
            with pd.HDFStore(self.path, mode="a") as hdf:
                return hdf.select(name, where=where) if name in hdf else pd.DataFrame(columns=["key", "value", "updated_at_utc"])
        if name not in TABLE_SPECS:
            raise KeyError(f"unknown table: {name}")
        with pd.HDFStore(self.path, mode="a") as hdf:
            if f"/{name}" not in hdf.keys():
                return pd.DataFrame(columns=TABLE_SPECS[name].columns)
            return hdf.select(name, where=where)

    def replace(self, name: str, frame: pd.DataFrame) -> None:
        normalized = normalize_table(name, frame)
        spec = TABLE_SPECS[name]
        with pd.HDFStore(self.path, mode="a") as hdf:
            hdf.put(name, normalized, format="table", data_columns=list(spec.data_columns), min_itemsize=spec.min_itemsize, index=False)

    def upsert(self, name: str, frame: pd.DataFrame) -> None:
        incoming = normalize_table(name, frame)
        existing = self.read(name)
        if existing.empty:
            self.replace(name, incoming)
            return
        spec = TABLE_SPECS[name]
        provenance = {"fetched_at_utc", "source_update_time", "updated_at_utc"}
        business = [c for c in spec.columns if c not in provenance]
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.drop_duplicates(spec.key, keep="first")
        for _, new_row in incoming.iterrows():
            mask = pd.Series(True, index=combined.index)
            for column in spec.key:
                mask &= combined[column] == new_row[column]
            if mask.any():
                old = combined.loc[mask, business].iloc[0]
                if not old.equals(new_row[business]):
                    combined.loc[mask, :] = new_row.values
        self.replace(name, combined)

    def read_metadata(self) -> dict[str, object]:
        frame = self.read("_metadata")
        return {row.key: json.loads(row.value) for row in frame.itertuples()}

    def write_metadata(self, updates: Mapping[str, object]) -> None:
        current = self.read_metadata()
        current.update(updates)
        frame = _metadata_frame(current)
        with pd.HDFStore(self.path, mode="a") as hdf:
            hdf.put("_metadata", frame, format="table", data_columns=["key"], min_itemsize={"key": 128, "value": 16384}, index=False)

    def keys(self) -> set[str]:
        with pd.HDFStore(self.path, mode="a") as hdf:
            return {key.lstrip("/") for key in hdf.keys()}


@contextmanager
def staged_store(active_path: Path, staging_path: Path, run_fingerprint: str, *, reset_staging: bool = False, lock_path: Path | None = None) -> Iterator[CryptoQuantStore]:
    active_path, staging_path = Path(active_path), Path(staging_path)
    lock_path = Path(lock_path) if lock_path is not None else active_path.with_suffix(active_path.suffix + ".lock")
    with single_writer_lock(lock_path):
        if staging_path.exists() and reset_staging:
            staging_path.unlink()
        if staging_path.exists():
            found = CryptoQuantStore(staging_path).read_metadata().get("run_fingerprint")
            if found != run_fingerprint:
                raise StagingMismatchError(f"staging fingerprint {found!r} != {run_fingerprint!r}")
        elif active_path.exists():
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(active_path, staging_path)
        else:
            staging_path.parent.mkdir(parents=True, exist_ok=True)
        store = CryptoQuantStore(staging_path)
        if store.read_metadata().get("run_fingerprint") != run_fingerprint:
            store.write_metadata({"run_fingerprint": run_fingerprint})
        try:
            yield store
        except Exception:
            raise
        else:
            os.replace(staging_path, active_path)
