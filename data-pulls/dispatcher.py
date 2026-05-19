"""
aRCADA data pull dispatcher.
Entry point for GitHub Actions fetch-data.yml.
Reads a JSON plan, routes each instrument to the correct pull script,
converts results to Zarr, and writes metadata.

Usage:
    python dispatcher.py --plan '{"instruments":[...], "time_range":{...}}' --out /tmp/arcada_out
    python dispatcher.py --plan-file plan.json --out /tmp/arcada_out
"""

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone

import xarray as xr

from earthscope import fetch_earthscope_instrument
from ooi_api import fetch_ooi_instrument
from pi_scraper import fetch_pi_instrument
from to_zarr import datasets_to_zarr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("arcada.dispatcher")

FETCHERS = {
    "ooi_api":     fetch_ooi_instrument,
    "earthscope":  fetch_earthscope_instrument,
    "pi_html":     fetch_pi_instrument,
}

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "catalog", "instruments.json")


def load_catalog() -> dict[str, dict]:
    with open(CATALOG_PATH) as f:
        catalog = json.load(f)
    return {inst["id"]: inst for inst in catalog["instruments"]}


def parse_time_range(tr: dict) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(tr["start"].replace("Z", "+00:00"))
    end   = datetime.fromisoformat(tr["end"].replace("Z", "+00:00"))
    return start.replace(tzinfo=timezone.utc), end.replace(tzinfo=timezone.utc)


def run(plan: dict, out_dir: str) -> dict:
    catalog = load_catalog()
    job_id  = plan.get("job_id") or str(uuid.uuid4())[:8]
    start, end = parse_time_range(plan["time_range"])

    log.info("Job %s | %d instruments | %s – %s", job_id, len(plan["instruments"]), start, end)

    tmp_dir = os.path.join(out_dir, "nc_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    datasets: dict[str, xr.Dataset] = {}
    errors: list[dict] = []

    for inst in plan["instruments"]:
        iid    = inst["id"]
        source = inst.get("source")

        # Merge catalog metadata into instrument spec (catalog is authoritative)
        cfg = {**catalog.get(iid, {}), **inst}

        if source not in FETCHERS:
            log.error("Unknown source '%s' for %s — skipping", source, iid)
            errors.append({"instrument_id": iid, "error": f"Unknown source: {source}"})
            continue

        log.info("Pulling %s via %s", iid, source)
        try:
            ds = FETCHERS[source](
                instrument_cfg=cfg,
                start=start,
                end=end,
                out_dir=tmp_dir,
                ooi_username=os.environ.get("OOI_USERNAME", ""),
                ooi_token=os.environ.get("OOI_TOKEN", ""),
            )
            if ds is not None and ds.data_vars:
                datasets[iid] = ds
            else:
                log.warning("Empty dataset returned for %s", iid)
                errors.append({"instrument_id": iid, "error": "Empty dataset"})
        except Exception as e:
            log.error("Failed to fetch %s: %s", iid, e, exc_info=True)
            errors.append({"instrument_id": iid, "error": str(e)})

    if not datasets:
        log.error("No data was successfully retrieved.")
        sys.exit(1)

    zarr_path, metadata = datasets_to_zarr(datasets, out_dir, job_id)
    metadata["errors"] = errors
    metadata["plan"]   = plan

    # Write final metadata again with errors included
    import json as _json
    with open(os.path.join(out_dir, f"arcada_{job_id}_metadata.json"), "w") as f:
        _json.dump(metadata, f, indent=2, default=str)

    log.info("Done. Zarr: %s | %d instruments | %d errors", zarr_path, len(datasets), len(errors))
    return metadata


def main():
    parser = argparse.ArgumentParser(description="aRCADA data pull dispatcher")
    parser.add_argument("--plan",      type=str, help="JSON plan string")
    parser.add_argument("--plan-file", type=str, help="Path to JSON plan file")
    parser.add_argument("--out",       type=str, default="/tmp/arcada_out", help="Output directory")
    args = parser.parse_args()

    if args.plan_file:
        with open(args.plan_file) as f:
            plan = json.load(f)
    elif args.plan:
        plan = json.loads(args.plan)
    else:
        # Also accept ARCADA_PLAN env var (set by GitHub Actions)
        plan_env = os.environ.get("ARCADA_PLAN")
        if plan_env:
            plan = json.loads(plan_env)
        else:
            parser.error("Provide --plan, --plan-file, or set ARCADA_PLAN env var")

    run(plan, args.out)


if __name__ == "__main__":
    main()
