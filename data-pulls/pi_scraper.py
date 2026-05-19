"""
PI instrument HTML page scraper for aRCADA.
Fetches data from PI-operated Apache directory listings at piweb.ooirsn.uw.edu.
Handles OVRSRA101 (scanning sonar .wc files), MASSP, and RASSP instruments.
"""

import io
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

NUM_RE      = re.compile(rb"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
DIR_TS_RE   = re.compile(r"(?P<Y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})[_T](?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})")
ENCODINGS   = ["utf-8", "latin-1", "utf-16-le", "utf-16-be"]
CHUNK_BYTES = 65536
MAX_BYTES   = 30 * 1024 * 1024  # 30 MB per file


def _get(url: str, stream: bool = False, timeout: int = 30) -> requests.Response:
    for attempt in range(4):
        try:
            r = requests.get(url, stream=stream, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
            log.warning("Retry %d for %s: %s", attempt + 1, url, e)


def _detect_encoding(raw_sample: bytes) -> str:
    """Score each candidate encoding by number of valid decoded chars."""
    best, best_score = "latin-1", -1
    for enc in ENCODINGS:
        try:
            text = raw_sample.decode(enc)
            score = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
            if score > best_score:
                best, best_score = enc, score
        except Exception:
            continue
    return best


def _list_apache_dir(url: str) -> list[str]:
    """Return all href entries from an Apache directory listing."""
    r = _get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True) if not a["href"].startswith("?")]


def _list_time_windows(base_url: str, start: datetime, end: datetime) -> list[dict]:
    """
    Walk the {YYYY}/{MM}/{timestamp}/data/ directory tree,
    returning file entries within [start, end].
    """
    entries = []
    # Iterate month-by-month
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        month_url = f"{base_url}{year:04d}/{month:02d}/"
        try:
            dirs = _list_apache_dir(month_url)
        except Exception as e:
            log.warning("Could not list %s: %s", month_url, e)
            _advance_month(year, month)
            year, month = _advance_month(year, month)
            continue

        for d in dirs:
            m = DIR_TS_RE.search(d)
            if not m:
                continue
            ts = datetime(int(m["Y"]), int(m["m"]), int(m["d"]),
                          int(m["H"]), int(m["M"]), int(m["S"]))
            if not (start <= ts <= end):
                continue
            data_url = f"{month_url}{d}data/" if not d.endswith("/") else f"{month_url}{d}data/"
            try:
                files = _list_apache_dir(data_url)
            except Exception:
                continue
            for fname in files:
                if fname.startswith(".") or fname == "Parent Directory":
                    continue
                entries.append({"time": ts, "filename": fname, "url": data_url + fname})

        year, month = _advance_month(year, month)
    return entries


def _advance_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _stream_numeric_file(url: str) -> Optional[np.ndarray]:
    """Stream a binary/text file from URL, extract all numbers via regex."""
    total = 0
    buffer = b""
    values: list[float] = []

    with _get(url, stream=True, timeout=60) as r:
        sample = r.raw.read(8192)
        if not sample:
            return None
        enc = _detect_encoding(sample)
        r.raw.seek(0)  # not always possible; re-request handled below

    # Re-request to stream from the beginning
    with _get(url, stream=True, timeout=60) as r:
        for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_BYTES:
                log.warning("File %s exceeds %dMB limit, truncating", url, MAX_BYTES // 1024 // 1024)
                break
            combined = buffer + chunk
            # Keep last 32 bytes as carry-over in case a number spans chunks
            matches = list(NUM_RE.finditer(combined[:-32]))
            buffer = combined[-32:]
            for m in matches:
                values.append(float(m.group()))

    # Flush buffer
    for m in NUM_RE.finditer(buffer):
        values.append(float(m.group()))

    return np.asarray(values, dtype=np.float32) if values else None


def fetch_pi_instrument(
    instrument_cfg: dict,
    start: datetime,
    end: datetime,
    out_dir: str,
    **_,
) -> xr.Dataset:
    """
    Entry point called by dispatcher.py for PI HTML instruments.
    Returns an xarray Dataset with per-burst statistics.
    """
    base_url = instrument_cfg["pi_base_url"]
    iid      = instrument_cfg["id"]
    fmt      = instrument_cfg.get("file_format", "binary_text_mixed")

    log.info("Scraping PI instrument %s from %s", iid, base_url)
    entries = _list_time_windows(base_url, start, end)
    log.info("Found %d files for %s in range", len(entries), iid)

    if not entries:
        log.warning("No PI data files found for %s in %s – %s", iid, start, end)
        return xr.Dataset()

    rows = []
    for entry in entries:
        if fmt == "csv":
            rows.extend(_parse_csv_entry(entry))
        else:
            stats = _parse_numeric_entry(entry)
            if stats:
                rows.append(stats)

    if not rows:
        return xr.Dataset()

    df = pd.DataFrame(rows).set_index("time").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)

    ds = xr.Dataset.from_dataframe(df)
    ds.attrs.update({
        "instrument_id":   iid,
        "instrument_name": instrument_cfg["name"],
        "pi_base_url":     base_url,
        "latitude":        instrument_cfg.get("latitude"),
        "longitude":       instrument_cfg.get("longitude"),
        "depth_m":         instrument_cfg.get("depth_m"),
        "source":          "pi_html",
        "fetch_start":     start.isoformat(),
        "fetch_end":       end.isoformat(),
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_files":         len(entries),
    })

    import os
    nc_path = os.path.join(out_dir, f"{iid.replace('/', '_')}.nc")
    ds.to_netcdf(nc_path)
    log.info("Saved %d records to %s", len(df), nc_path)
    return ds


def _parse_numeric_entry(entry: dict) -> Optional[dict]:
    """Stream a binary/text file and return per-burst summary statistics."""
    vals = _stream_numeric_file(entry["url"])
    if vals is None or len(vals) == 0:
        return None
    return {
        "time":   entry["time"],
        "mean":   float(np.mean(vals)),
        "median": float(np.median(vals)),
        "max":    float(np.max(vals)),
        "min":    float(np.min(vals)),
        "p95":    float(np.percentile(vals, 95)),
        "n":      len(vals),
        "url":    entry["url"],
    }


def _parse_csv_entry(entry: dict) -> list[dict]:
    """Fetch and parse a CSV file from a PI instrument page."""
    try:
        r = _get(entry["url"], timeout=30)
        df = pd.read_csv(io.StringIO(r.text))
        df["source_url"] = entry["url"]
        return df.to_dict(orient="records")
    except Exception as e:
        log.warning("Failed to parse CSV %s: %s", entry["url"], e)
        return []
