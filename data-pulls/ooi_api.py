"""
OOI M2M API data pull for aRCADA.
Handles all instruments served through the OOI REST API:
  pressure (BOTPT), CTD, hydrophone, pCO2, thermistor, OBS.
"""

import os
import re
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr

log = logging.getLogger(__name__)

OOI_BASE = "https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv"
NUM_RE   = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Stream definitions keyed by instrument type
STREAM_MAP = {
    "pressure":     "botpt_nano_sample",
    "ctd":          "ctdpf_ckl_wfp_instrument",
    "hydrophone":   "hydrophone_a_dcl_instrument",
    "pco2":         "pco2w_a_dcl_instrument",
    "thermistor":   "thsph_a_dcl_instrument",
    "seismometer":  "obsbb_a_dcl_instrument",
}


def _get(url: str, auth: tuple[str, str] | None, **kwargs) -> requests.Response:
    """GET with retry and exponential backoff."""
    for attempt in range(4):
        try:
            r = requests.get(url, auth=auth, timeout=60, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == 3:
                raise
            wait = 2 ** attempt
            log.warning("Attempt %d failed (%s), retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)


def request_async_delivery(
    site: str,
    node: str,
    instrument: str,
    stream: str,
    start: datetime,
    end: datetime,
    username: str,
    token: str,
) -> dict:
    """
    Submit an async data delivery request to the OOI M2M API.
    Returns the response JSON containing the request status URL.
    """
    url = f"{OOI_BASE}/{site}/{node}/{instrument}/{stream}"
    params = {
        "beginDT": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "endDT":   end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "format":  "application/netcdf",
        "limit":   -1,
        "execDPA": True,
    }
    r = _get(url, auth=(username, token), params=params)
    return r.json()


def poll_async_request(status_url: str, username: str, token: str, timeout_s: int = 600) -> list[str]:
    """
    Poll an OOI async request until complete, return list of download URLs.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = _get(status_url, auth=(username, token))
        data = r.json()
        if data.get("status") == "complete":
            return data.get("allURLs", [])
        if data.get("status") == "failed":
            raise RuntimeError(f"OOI async request failed: {data}")
        time.sleep(15)
    raise TimeoutError(f"OOI async request did not complete within {timeout_s}s")


def download_netcdf(url: str, out_path: str, username: str, token: str) -> str:
    """Stream a NetCDF file from OOI thredds to disk."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with _get(url, auth=(username, token), stream=True) as r:
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    return out_path


def fetch_ooi_instrument(
    instrument_cfg: dict,
    start: datetime,
    end: datetime,
    out_dir: str,
    ooi_username: str = "",
    ooi_token: str = "",
) -> xr.Dataset:
    """
    High-level: fetch one OOI API instrument for a time range.
    Returns an xarray Dataset. Writes intermediate NetCDF to out_dir.

    For public instruments (no auth needed), pass empty strings.
    For M2M auth, pass OOI username + API token.
    """
    site       = instrument_cfg["site"]
    node       = instrument_cfg["node"]
    instrument = instrument_cfg["instrument"]
    itype      = instrument_cfg["type"]
    stream     = instrument_cfg.get("stream") or STREAM_MAP.get(itype, "")

    if not stream:
        raise ValueError(f"No stream defined for instrument type '{itype}'")

    log.info("Requesting OOI data: %s/%s/%s stream=%s", site, node, instrument, stream)

    auth = (ooi_username, ooi_token) if ooi_username else None

    # Direct synchronous endpoint (works for small requests)
    url = (
        f"{OOI_BASE}/{site}/{node}/{instrument}/{stream}"
        f"?beginDT={start.strftime('%Y-%m-%dT%H:%M:%S.000Z')}"
        f"&endDT={end.strftime('%Y-%m-%dT%H:%M:%S.000Z')}"
        f"&format=application/json&limit=20000"
    )

    r = _get(url, auth=auth)
    data = r.json()

    if not data:
        log.warning("OOI returned empty dataset for %s", instrument_cfg["id"])
        return xr.Dataset()

    # Parse JSON response into xarray Dataset
    df = pd.DataFrame(data)
    if "time" not in df.columns:
        log.warning("No 'time' column in OOI response for %s", instrument_cfg["id"])
        return xr.Dataset()

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()

    # Drop non-numeric columns (annotations, QC flags as strings)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]

    ds = xr.Dataset.from_dataframe(df)
    ds.attrs.update({
        "instrument_id":   instrument_cfg["id"],
        "instrument_name": instrument_cfg["name"],
        "site":            site,
        "node":            node,
        "instrument":      instrument,
        "stream":          stream,
        "latitude":        instrument_cfg.get("latitude"),
        "longitude":       instrument_cfg.get("longitude"),
        "depth_m":         instrument_cfg.get("depth_m"),
        "source":          "ooi_api",
        "fetch_start":     start.isoformat(),
        "fetch_end":       end.isoformat(),
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
    })

    nc_path = os.path.join(out_dir, f"{instrument_cfg['id'].replace('/', '_')}.nc")
    ds.to_netcdf(nc_path)
    log.info("Saved NetCDF: %s", nc_path)

    return ds


def check_data_gaps(ds: xr.Dataset, expected_freq_s: Optional[float] = None) -> list[dict]:
    """
    Identify gaps in a time series Dataset.
    Returns list of {start, end, duration_s} dicts.
    """
    if "time" not in ds.dims or len(ds.time) < 2:
        return []

    times = pd.DatetimeIndex(ds.time.values)
    diffs = times[1:] - times[:-1]

    if expected_freq_s is None:
        expected_freq_s = float(np.median(diffs.total_seconds()))

    threshold = expected_freq_s * 5  # gap = 5x nominal interval
    gaps = []
    for i, d in enumerate(diffs):
        if d.total_seconds() > threshold:
            gaps.append({
                "start":      times[i].isoformat(),
                "end":        times[i + 1].isoformat(),
                "duration_s": d.total_seconds(),
            })
    return gaps
