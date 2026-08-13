import logging
import os
import zipfile
import tempfile
import shutil
import time
from pathlib import Path
from typing import Optional

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

ERA5_SOLAR_INSTANTANEOUS_VARS = [
    "2m_temperature", 
    "2m_dewpoint_temperature", 
    "10m_u_component_of_wind", 
    "10m_v_component_of_wind", 
    "surface_pressure", 
    "total_cloud_cover",  
    "total_column_water_vapour",
]

ERA5_ACCUMULATED_VARS = [
    "surface_solar_radiation_downwards",  # ssrd 
    "total_sky_direct_solar_radiation_at_surface", # fdir - direct beam
    "total_precipitation",                # tp — useful for weather features too 
]

ERA5_RESOLUTION_DEG = 0.25
ERA5_LAT_START      = 90.0
ERA5_LON_START      = -180.0

def index_to_lat(i: int) -> float:
    return ERA5_LAT_START - i * ERA5_RESOLUTION_DEG


def index_to_lon(j: int) -> float:
    return ERA5_LON_START + j * ERA5_RESOLUTION_DEG
def _compute_registry_bbox(registry_df: pd.DataFrame, pad: float = ERA5_RESOLUTION_DEG) -> list[float]:
    """
    Compute [N, W, S, E] bounding box covering all plant cells in the registry,
    padded by `pad` degrees (default 1 cell) so every 3×3 patch is fully covered.
    """
    lats = registry_df["era5_i"].map(lambda i: index_to_lat(i))
    lons = registry_df["era5_j"].map(lambda j: index_to_lon(j))
    return [
        round(float(lats.max()) + pad, 3),  # N
        round(float(lons.min()) - pad, 3),  # W
        round(float(lats.min()) - pad, 3),  # S
        round(float(lons.max()) + pad, 3),  # E
    ]


def _download_nc(client: cdsapi.Client, params: dict, out_path: Path) -> Path:
    """
    Submit one CDS request, handle zip vs bare .nc response, write to out_path.
    Uses a temp file so a crash mid-download never leaves a partial cache hit.
    """
    tmp_path = out_path.with_suffix(".tmp")
    try:
        client.retrieve("reanalysis-era5-single-levels", params).download(str(tmp_path))

        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path) as zf:
                nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
                if not nc_members:
                    raise ValueError(f"No .nc found inside zip for {out_path.name}")
                extracted = zf.extract(nc_members[0], path=out_path.parent)
                shutil.move(extracted, out_path)
            tmp_path.unlink()
        else:
            tmp_path.rename(out_path)

    except Exception:
        log.info('Request failed')
        tmp_path.unlink(missing_ok=True)  # never leave .tmp as a phantom cache hit
        raise

    return out_path


# ── Core download: one month × one var group × full bbox ─────────────────────

def fetch_era5_bbox_monthly(
    area: list[float],
    year: int,
    month: int,
    var_suffix: str,
    var_list: list[str],
    cache_dir: Path,
    client: cdsapi.Client,
    sleep_secs: int = 10,
) -> Path:
    """
    Download one calendar month of ERA5 for the full plant bbox.

    Parameters
    ----------
    area        : [N, W, S, E] bounding box from _compute_registry_bbox()
    var_suffix  : short label used in the filename, e.g. "inst" or "accum"
    sleep_secs  : polite pause after each successful download
    """
    out_path = cache_dir / f"era5_{var_suffix}_{year}_{month:02d}.nc"
    if out_path.exists():
        log.info("Cache hit: %s", out_path.name)
        return out_path

    days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month
    log.info("CDS fetch: %d-%02d [%s]", year, month, var_suffix)

    _download_nc(
        client,
        {
            "product_type": "reanalysis",
            "variable":     var_list,
            "year":         str(year),
            "month":        f"{month:02d}",
            "day":          [f"{d:02d}" for d in range(1, days_in_month + 1)],
            "time":         [f"{h:02d}:00" for h in range(24)],
            "area":         area,
            "format":       "netcdf",
        },
        out_path,
    )

    time.sleep(sleep_secs)
    return out_path


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_era5_hourly_for_plants(
    registry_df: pd.DataFrame | None,
    bbox: list[float] | None,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str | Path] = None,
    client: Optional[cdsapi.Client] = None,
    extract_per_plant_patches: bool = False,
) -> dict[tuple[int, int], Path]:
    """
    Batch-fetch hourly ERA5 for all unique ERA5 grid cells in the registry.

    Strategy
    --------
    1. Compute a single bbox covering every plant (+ 1-cell pad for 3×3 patches).
    2. Download one .nc per (month × var_group) — serial, with cache checks.
    3. If extract_per_plant_patches is True, open all monthly files lazily, extract a 
       3×3 patch per unique cell, write per-cell .nc files to cache_dir.

    Returns
    -------
    dict mapping (era5_i, era5_j) → Path of the per-cell .nc file.
    """
    if cache_dir is None:
        cache_dir = Path(DATA_RAW_DIR) / "era5_hourly"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = client or cdsapi.Client()

    area = bbox if bbox is not None else _compute_registry_bbox(registry_df)
    log.info("Registry bbox [N=%.3f W=%.3f S=%.3f E=%.3f]", *area)

    months = pd.date_range(
        pd.Timestamp(start_date).replace(day=1),
        pd.Timestamp(end_date) + pd.offsets.MonthEnd(0),
        freq="MS",
    )
    # Drop the months we are missing data for.
    target_months = [5, 6, 7, 8, 9]
    months = months[months.month.isin(target_months)]

    ######## Phase 1 & 2: download all months (2 requests per month, serial) and save to netcdf #####
    inst_paths, accum_paths = [], []
    for month_start in months:
        y, m = month_start.year, month_start.month
        inst_paths.append(
            fetch_era5_bbox_monthly(area, y, m, "inst", ERA5_SOLAR_INSTANTANEOUS_VARS, cache_dir, client)
        )
        accum_paths.append(
            fetch_era5_bbox_monthly(area, y, m, "accum", ERA5_ACCUMULATED_VARS, cache_dir, client)
        )

    ######### Phase 3: Optionally extract 3x3 patches surrounding each plant. ###################
    if not extract_per_plant_patches:
    	log.info("All months downloaded - returning...")
    	return
    # ── Phase 2: extract per-cell 3×3 patches from the merged dataset ────────
    log.info("All months downloaded — extracting per-cell patches...")
    ds = xr.merge([
        xr.open_mfdataset(inst_paths,  combine="by_coords"),
        xr.open_mfdataset(accum_paths, combine="by_coords"),
    ])
    time_coord = "valid_time" if "valid_time" in ds.dims else "time"

    unique_cells = list(
        registry_df[["era5_i", "era5_j"]].drop_duplicates().itertuples(index=False)
    )

    cell_paths: dict[tuple[int, int], Path] = {}
    for cell in unique_cells:
        i, j = int(cell.era5_i), int(cell.era5_j)
        out_path = cache_dir / f"era5h_i{i}_j{j}_{start_date}_{end_date}.nc"

        if out_path.exists():
            log.info("Patch cache hit: %s", out_path.name)
            cell_paths[(i, j)] = out_path
            continue

        try:
            patch = extract_3x3_patch(ds, i, j)
            patch.to_netcdf(out_path)
            cell_paths[(i, j)] = out_path
            log.info("Wrote patch: %s", out_path.name)
        except Exception as e:
            log.warning("Failed to extract cell (%d, %d): %s", i, j, e)

    ds.close()
    return cell_paths

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

	#registry = pd.read_parquet("data/processed/plant_registry.parquet")

	# See how many unique cells you have total
	#unique_cells = registry[["era5_i", "era5_j"]].drop_duplicates()
	#print(f"{len(unique_cells)} unique ERA5 cells for {len(registry)} plants")
	# bbox for registry is [N=49.000 W=-123.750 S=31.500 E=-104.250]
	bbox = [42.000, -124.500, 31.750, -114.000]
	cell_paths = fetch_era5_hourly_for_plants(
	    registry_df=None,
	    bbox=bbox,
	    start_date="2023-01-01",
	    end_date="2026-01-01",
	    cache_dir="data/raw/ca_2023_2026_era5_hourly",
        extract_per_plant_patches=False,
	)