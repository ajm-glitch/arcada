"""
Convert fetched xarray Datasets to Zarr format and generate metadata JSON.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import zarr

log = logging.getLogger(__name__)


def datasets_to_zarr(
    datasets: dict[str, xr.Dataset],
    out_dir: str,
    job_id: str,
) -> tuple[str, dict]:
    """
    Write a dict of {instrument_id: xr.Dataset} to a single Zarr store.
    Each instrument becomes a group inside the store.
    Returns (zarr_path, metadata_dict).
    """
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, f"arcada_{job_id}.zarr")

    store = zarr.open_group(zarr_path, mode="w")

    metadata = {
        "job_id":          job_id,
        "created":         datetime.now(timezone.utc).isoformat(),
        "format":          "zarr",
        "zarr_version":    "2",
        "instruments":     [],
        "time_range":      {"start": None, "end": None},
        "total_variables": 0,
        "total_records":   0,
    }

    all_starts, all_ends = [], []

    for iid, ds in datasets.items():
        if ds is None or not ds.data_vars:
            log.warning("Skipping empty dataset for %s", iid)
            continue

        safe_id = iid.replace("/", "_").replace("-", "_").replace(".", "_")
        grp = store.require_group(safe_id)

        # Write each variable to Zarr
        for var in ds.data_vars:
            arr = ds[var].values
            chunks = _auto_chunks(arr.shape)
            grp.create_dataset(var, data=arr, chunks=chunks, dtype=arr.dtype, overwrite=True)

        # Write time coordinate
        if "time" in ds.coords:
            times = ds.coords["time"].values
            time_unix = pd.DatetimeIndex(times).astype(np.int64) // 1_000_000_000
            grp.create_dataset("time", data=time_unix, overwrite=True)
            grp.attrs["time_units"] = "seconds since 1970-01-01T00:00:00Z"
            t_start = pd.Timestamp(times[0]).isoformat()
            t_end   = pd.Timestamp(times[-1]).isoformat()
            all_starts.append(t_start)
            all_ends.append(t_end)
        else:
            t_start = t_end = None

        # Copy dataset attributes to group
        grp.attrs.update(ds.attrs)

        gaps = _detect_gaps(ds)

        instrument_meta = {
            "instrument_id":   iid,
            "instrument_name": ds.attrs.get("instrument_name", iid),
            "type":            ds.attrs.get("type", "unknown"),
            "source":          ds.attrs.get("source", "unknown"),
            "latitude":        ds.attrs.get("latitude"),
            "longitude":       ds.attrs.get("longitude"),
            "depth_m":         ds.attrs.get("depth_m"),
            "variables":       list(ds.data_vars),
            "n_records":       int(ds.dims.get("time", 0)),
            "coverage_start":  t_start,
            "coverage_end":    t_end,
            "gaps":            gaps,
            "units":           _extract_units(ds),
            "zarr_group":      safe_id,
        }
        metadata["instruments"].append(instrument_meta)
        metadata["total_records"] += instrument_meta["n_records"]
        metadata["total_variables"] += len(ds.data_vars)

    if all_starts:
        metadata["time_range"]["start"] = min(all_starts)
        metadata["time_range"]["end"]   = max(all_ends)

    meta_path = os.path.join(out_dir, f"arcada_{job_id}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info("Zarr store: %s | Metadata: %s", zarr_path, meta_path)
    return zarr_path, metadata


def _auto_chunks(shape: tuple) -> tuple:
    """Heuristic chunking: ~1MB chunks for 1D, smaller for higher dims."""
    if len(shape) == 1:
        return (min(shape[0], 100_000),)
    return tuple(min(d, 1000) for d in shape)


def _detect_gaps(ds: xr.Dataset, threshold_multiplier: float = 5.0) -> list[dict]:
    """Return list of gap dicts {start, end, duration_s} for a time series."""
    if "time" not in ds.dims or len(ds.time) < 2:
        return []
    times = pd.DatetimeIndex(ds.time.values)
    diffs = (times[1:] - times[:-1]).total_seconds()
    median_dt = float(np.median(diffs))
    threshold = median_dt * threshold_multiplier
    gaps = []
    for i, d in enumerate(diffs):
        if d > threshold:
            gaps.append({
                "start":      times[i].isoformat(),
                "end":        times[i + 1].isoformat(),
                "duration_s": d,
            })
    return gaps


def _extract_units(ds: xr.Dataset) -> dict[str, Optional[str]]:
    """Extract unit attributes from each variable."""
    units = {}
    for var in ds.data_vars:
        da = ds[var]
        units[var] = da.attrs.get("units") or da.attrs.get("unit")
    return units
