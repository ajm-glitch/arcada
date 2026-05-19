"""
EarthScope/IRIS FDSN seismic data pull for aRCADA.
Uses ObsPy's FDSN client to retrieve waveforms and metadata from EarthScope
for OOI Regional Cabled Array seismic stations (network OO).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

log = logging.getLogger(__name__)

FDSN_CLIENT = "IRIS"
DEFAULT_NETWORK = "OO"

# Channel priority order: prefer broadband, fall back to short-period
CHANNEL_PRIORITY = ["BH*", "HH*", "EH*", "SH*"]

# Known RCA seismic stations
RCA_STATIONS = {
    "AXCC1": {"lat": 45.9549, "lon": -130.0089, "depth_m": 1519, "location": "Axial Caldera Center"},
    "AXEC1": {"lat": 45.9420, "lon": -130.0148, "depth_m": 1519, "location": "Axial East Caldera 1"},
    "AXEC2": {"lat": 45.9399, "lon": -129.9739, "depth_m": 1519, "location": "Axial East Caldera 2"},
    "AXEC3": {"lat": 45.9173, "lon": -129.9754, "depth_m": 1519, "location": "Axial East Caldera 3"},
    "AXID1": {"lat": 45.9234, "lon": -130.0147, "depth_m": 1560, "location": "Axial Int'l District"},
    **{f"HYS{str(i).zfill(2)}": {"lat": 44.5692, "lon": -125.1479, "depth_m": 775, "location": "Hydrate Ridge"}
       for i in range(1, 15)},
}


def _utc(dt: datetime) -> UTCDateTime:
    return UTCDateTime(dt.replace(tzinfo=timezone.utc).timestamp())


def fetch_inventory(
    network: str = DEFAULT_NETWORK,
    station: str = "*",
    channel: str = "BH*,HH*",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> dict:
    """Return station inventory metadata as a plain dict."""
    client = Client(FDSN_CLIENT)
    inv = client.get_stations(
        network=network,
        station=station,
        channel=channel,
        starttime=_utc(start) if start else None,
        endtime=_utc(end) if end else None,
        level="channel",
    )
    stations = []
    for net in inv:
        for sta in net:
            for cha in sta:
                stations.append({
                    "network":   net.code,
                    "station":   sta.code,
                    "channel":   cha.code,
                    "location":  cha.location_code,
                    "latitude":  sta.latitude,
                    "longitude": sta.longitude,
                    "elevation": sta.elevation,
                    "sample_rate": cha.sample_rate,
                    "start_date": str(cha.start_date),
                    "end_date":   str(cha.end_date) if cha.end_date else None,
                })
    return stations


def fetch_waveforms(
    instrument_cfg: dict,
    start: datetime,
    end: datetime,
    channel: str = "BH*",
    apply_filter: bool = True,
    filter_freq_hz: float = 1.0,
) -> xr.Dataset:
    """
    Fetch seismic waveforms for a single station over [start, end].
    Returns an xarray Dataset with one variable per channel (Z, N, E).
    Large time ranges are fetched day-by-day to manage memory.
    """
    network = instrument_cfg.get("network", DEFAULT_NETWORK)
    station = instrument_cfg["station"]

    client = Client(FDSN_CLIENT)
    log.info("Fetching waveforms: %s.%s %s %s – %s", network, station, channel, start, end)

    all_arrays: dict[str, list] = {}
    all_times: list = []

    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=1), end)
        try:
            st = client.get_waveforms(
                network=network,
                station=station,
                location="*",
                channel=channel,
                starttime=_utc(current),
                endtime=_utc(chunk_end),
            )
        except Exception as e:
            log.warning("No data for %s %s – %s: %s", station, current, chunk_end, e)
            current = chunk_end
            continue

        # Quality filtering
        st = _filter_stream(st, apply_filter, filter_freq_hz)

        for tr in st:
            ch = tr.stats.channel
            times_sec = tr.times("timestamp")  # Unix timestamps
            if ch not in all_arrays:
                all_arrays[ch] = []
            all_arrays[ch].append((times_sec, tr.data.astype(np.float32)))

        current = chunk_end

    if not all_arrays:
        log.warning("No waveform data returned for %s", station)
        return xr.Dataset()

    # Concatenate per-channel and build Dataset
    data_vars = {}
    time_index = None
    for ch, chunks in all_arrays.items():
        times_cat = np.concatenate([c[0] for c in chunks])
        data_cat  = np.concatenate([c[1] for c in chunks])
        if time_index is None:
            time_index = pd.to_datetime(times_cat, unit="s", utc=True)
        data_vars[ch] = xr.DataArray(data_cat, dims=["time"])

    ds = xr.Dataset(data_vars, coords={"time": time_index})
    ds.attrs.update({
        "instrument_id":   instrument_cfg["id"],
        "instrument_name": instrument_cfg["name"],
        "network":         network,
        "station":         station,
        "channel":         channel,
        "latitude":        instrument_cfg.get("latitude"),
        "longitude":       instrument_cfg.get("longitude"),
        "depth_m":         instrument_cfg.get("depth_m"),
        "source":          "earthscope",
        "filter_applied":  apply_filter,
        "filter_freq_hz":  filter_freq_hz if apply_filter else None,
        "fetch_start":     start.isoformat(),
        "fetch_end":       end.isoformat(),
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
    })

    log.info("Fetched %d samples across %d channels for %s", len(time_index), len(data_vars), station)
    return ds


def _filter_stream(st, apply_filter: bool, freq_hz: float):
    """Detrend, taper, and optionally highpass filter an ObsPy Stream."""
    st.merge(method=1, fill_value=0)
    st = st.select(sampling_rate=st[0].stats.sampling_rate if st else 100.0)
    for tr in st:
        if tr.data is None or len(tr.data) == 0:
            st.remove(tr)
            continue
        if np.all(tr.data == 0) or np.all(np.isnan(tr.data.astype(float))):
            st.remove(tr)
            continue
        tr.detrend(type="linear")
        tr.taper(type="hann", max_percentage=0.05)
        if apply_filter and freq_hz > 0:
            tr.filter(type="highpass", freq=freq_hz)
    return st


def fetch_earthscope_instrument(
    instrument_cfg: dict,
    start: datetime,
    end: datetime,
    out_dir: str,
    **kwargs,
) -> xr.Dataset:
    """Entry point called by dispatcher.py."""
    ds = fetch_waveforms(instrument_cfg, start, end, **kwargs)
    if ds:
        import os
        nc_path = os.path.join(out_dir, f"{instrument_cfg['id'].replace('/', '_')}.nc")
        ds.to_netcdf(nc_path)
        log.info("Saved NetCDF: %s", nc_path)
    return ds
