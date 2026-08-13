"""
ERA5 + CAISO LMP Alignment Pipeline
=====================================
Outputs two artifacts:
  1. caiso_lmp_era5_averaged.parquet  — zone-averaged weather, ready for iTransformer baseline
  2. era5_spatial_<zone>.zarr         — raw ERA5 grid per zone, ready for spatiotemporal arch

Usage:
    python era5_lmp_alignment.py

Requirements:
    pip install xarray zarr netcdf4 pandas pyarrow tqdm
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

ERA5_INST_GLOB  = "data/raw/ca_2023_2026_era5_hourly/era5_inst_*.nc"
ERA5_ACCUM_GLOB = "data/raw/ca_2023_2026_era5_hourly/era5_accum_*.nc"
LMP_WIDE_PATH   = "data/processed/caiso_lmp_wide.parquet"
OUTPUT_DIR      = Path("data/aligned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CAISO zone bounding boxes [N, S, W, E]
ZONE_BOXES = {
    "NP15": {"lat": (36.5, 42.0), "lon": (-124.5, -119.5)},
    "SP15": {"lat": (32.5, 36.5), "lon": (-120.5, -114.5)},
    "ZP26": {"lat": (35.0, 37.5), "lon": (-121.0, -118.0)},
}

# GRIB short name → clean name mapping
VAR_RENAME = {
    "t2m":  "t2m",
    "d2m":  "d2m",
    "u10":  "u10",
    "v10":  "v10",
    "sp":   "sp",
    "tcc":  "tcc",
    "tcwv": "tcwv",
    "ssrd": "ssrd",
    "fdir": "fdir",
    "tp":   "tp",
}

INSTANT_VARS  = ["t2m", "d2m", "u10", "v10", "sp", "tcc", "tcwv"]
ACCUM_VARS    = ["ssrd", "fdir", "tp"]
ALL_VARS      = INSTANT_VARS + ACCUM_VARS

# ── LOAD ERA5 ─────────────────────────────────────────────────────────────────

def load_era5(inst_glob: str, accum_glob: str) -> xr.Dataset:
    """
    Load and merge all monthly inst + accum NetCDF files.
    Returns a single Dataset with dim 'valid_time' in UTC (tz-naive).
    """
    inst_files  = sorted(glob.glob(inst_glob))
    accum_files = sorted(glob.glob(accum_glob))

    if not inst_files:
        raise FileNotFoundError(f"No files matched: {inst_glob}")
    if not accum_files:
        raise FileNotFoundError(f"No files matched: {accum_glob}")

    log.info(f"Loading {len(inst_files)} inst files, {len(accum_files)} accum files...")

    ds_inst  = xr.open_mfdataset(inst_files,  combine="by_coords", engine="netcdf4")
    ds_accum = xr.open_mfdataset(accum_files, combine="by_coords", engine="netcdf4")

    # Merge on shared time + lat/lon
    ds = xr.merge([ds_inst, ds_accum], compat="override")

    # Standardize time dim name (cfgrib uses 'valid_time', plain netcdf uses 'time')
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # Drop scalar coords that cause issues downstream
    scalar_coords = [c for c in ds.coords if ds[c].dims == ()]
    ds = ds.drop_vars(scalar_coords, errors="ignore")

    log.info(f"ERA5 loaded: {len(ds.time)} hours, "
             f"lat {float(ds.latitude.min()):.2f}–{float(ds.latitude.max()):.2f}, "
             f"lon {float(ds.longitude.min()):.2f}–{float(ds.longitude.max()):.2f}")
    return ds


# ── SPATIAL CROP PER ZONE ─────────────────────────────────────────────────────

def crop_zone(ds: xr.Dataset, zone: str) -> xr.Dataset:
    """
    Crop ERA5 to zone bounding box.
    ERA5 latitude decreases north→south, longitude increases west→east.
    """
    box = ZONE_BOXES[zone]
    lat_lo, lat_hi = box["lat"]
    lon_lo, lon_hi = box["lon"]

    # ERA5 lat dim is descending → use slice(hi, lo)
    ds_zone = ds.sel(
        latitude=slice(lat_hi, lat_lo),
        longitude=slice(lon_lo, lon_hi),
    )

    n_cells = len(ds_zone.latitude) * len(ds_zone.longitude)
    log.info(f"  {zone}: {len(ds_zone.latitude)} lat × {len(ds_zone.longitude)} lon = {n_cells} cells")
    return ds_zone


# ── VERSION 1: ZONE-AVERAGED PARQUET ──────────────────────────────────────────

def make_zone_averaged(ds: xr.Dataset) -> pd.DataFrame:
    """
    Spatial average over each zone box → one scalar per (variable, zone, hour).
    Returns wide DataFrame: one row per hour, columns = variable_ZONE.

    e.g. t2m_NP15, ssrd_SP15, u10_ZP26, ...
    """
    zone_dfs = []

    for zone in ZONE_BOXES:
        log.info(f"Averaging zone {zone}...")
        ds_zone = crop_zone(ds, zone)

        # Spatial mean over lat/lon → (time,) per variable
        zone_mean = ds_zone.mean(dim=["latitude", "longitude"])

        df = zone_mean.to_dataframe()[ALL_VARS].copy()
        df.index.name = "time"
        df.reset_index(inplace=True)

        # Rename: t2m → t2m_NP15
        df.rename(columns={v: f"{v}_{zone}" for v in ALL_VARS}, inplace=True)
        zone_dfs.append(df)

    # Merge all zones on time
    era5_wide = zone_dfs[0]
    for df in zone_dfs[1:]:
        era5_wide = era5_wide.merge(df, on="time", how="inner")

    # Add derived features useful for LMP
    for zone in ZONE_BOXES:
        t   = era5_wide[f"t2m_{zone}"]
        td  = era5_wide[f"d2m_{zone}"]
        # Relative humidity proxy (Magnus approximation)
        era5_wide[f"rh_{zone}"] = 100 * np.exp(
            17.625 * (td - 273.15) / (243.04 + td - 273.15) -
            17.625 * (t  - 273.15) / (243.04 + t  - 273.15)
        )
        # Wind speed scalar
        era5_wide[f"wind_speed_{zone}"] = np.sqrt(
            era5_wide[f"u10_{zone}"]**2 + era5_wide[f"v10_{zone}"]**2
        )

    log.info(f"Zone-averaged ERA5: {len(era5_wide)} rows, {len(era5_wide.columns)} columns")
    return era5_wide


# ── VERSION 2: SPATIAL GRID ZARR PER ZONE ─────────────────────────────────────

def make_spatial_zarr(ds: xr.Dataset) -> dict[str, Path]:
    """
    Save one zarr store per zone with full spatial grid preserved.
    Shape: (time, lat, lon, variables) via DataArray stack.

    These are the inputs for the spatiotemporal transformer's spatial encoder.
    """
    output_paths = {}

    for zone in ZONE_BOXES:
        out_path = OUTPUT_DIR / f"era5_spatial_{zone}.zarr"

        if out_path.exists():
            log.info(f"  Zarr cache hit: {out_path.name}")
            output_paths[zone] = out_path
            continue

        log.info(f"Writing spatial zarr for {zone}...")
        ds_zone = crop_zone(ds, zone)

        # Keep only the variables we need
        ds_zone = ds_zone[ALL_VARS]

        # Rechunk for efficient time-slicing (model will iterate over time windows)
        ds_zone = ds_zone.chunk({
            "time": 168,          # one lookback window per chunk
            "latitude": -1,       # full spatial dim in memory
            "longitude": -1,
        })

        ds_zone.to_zarr(out_path, mode="w")
        log.info(f"  Saved: {out_path} "
                 f"({ds_zone.dims['latitude']}×{ds_zone.dims['longitude']} cells)")
        output_paths[zone] = out_path

    return output_paths


# ── ALIGN ERA5 WITH LMP ────────────────────────────────────────────────────────

def align_with_lmp(era5_wide: pd.DataFrame, lmp_path: str) -> pd.DataFrame:
    """
    Timezone-convert ERA5 UTC → US/Pacific, then inner-join with LMP wide parquet.
    """
    # ERA5 time is UTC tz-naive → localize then convert
    era5_wide["time"] = (
        pd.to_datetime(era5_wide["time"], utc=True)
          .dt.tz_convert("US/Pacific")
    )

    # Load LMP
    lmp = pd.read_parquet(lmp_path)
    lmp["time"] = pd.to_datetime(lmp["time"])
    if lmp["time"].dt.tz is None:
        lmp["time"] = lmp["time"].dt.tz_localize("US/Pacific")

    n_before = len(era5_wide)
    merged = lmp.merge(era5_wide, on="time", how="inner")
    n_after = len(merged)

    log.info(f"LMP rows: {len(lmp):,}  ERA5 rows: {n_before:,}  "
             f"Merged: {n_after:,} ({n_after/len(lmp)*100:.1f}% overlap)")

    if n_after < 0.8 * len(lmp):
        log.warning("⚠️  <80% overlap — check timezone alignment or date range mismatch")

    return merged


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("ERA5 + CAISO LMP Alignment Pipeline")
    log.info("=" * 60)

    # ── Load ERA5 ────────────────────────────────────────────────────────────
    ds = load_era5(ERA5_INST_GLOB, ERA5_ACCUM_GLOB)

    # ── Version 1: Zone-averaged parquet ─────────────────────────────────────
    log.info("\n── Version 1: Zone-averaged (iTransformer baseline) ──")
    era5_wide   = make_zone_averaged(ds)
    merged      = align_with_lmp(era5_wide, LMP_WIDE_PATH)

    avg_path = OUTPUT_DIR / "caiso_lmp_era5_averaged.parquet"
    merged.to_parquet(avg_path, index=False)
    log.info(f"✓ Saved: {avg_path}")
    log.info(f"  Shape: {merged.shape}")
    log.info(f"  Columns: {[c for c in merged.columns if 't2m' in c or 'Cong' in c]}")

    # ── Null check ───────────────────────────────────────────────────────────
    null_cols = merged.columns[merged.isnull().any()].tolist()
    if null_cols:
        log.warning(f"⚠️  Null values in: {null_cols}")
        log.warning(merged[null_cols].isnull().mean().round(4).to_string())
    else:
        log.info("  ✓ No null values")

    # ── Version 2: Spatial zarr per zone ─────────────────────────────────────
    log.info("\n── Version 2: Spatial grid zarr (spatiotemporal arch) ──")
    zarr_paths = make_spatial_zarr(ds)
    for zone, path in zarr_paths.items():
        z = xr.open_zarr(path)
        log.info(f"  {zone}: {dict(z.dims)} — variables: {list(z.data_vars)}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("✓ Done. Outputs:")
    log.info(f"  Averaged parquet : {avg_path}")
    for zone, path in zarr_paths.items():
        log.info(f"  Spatial zarr ({zone}): {path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()