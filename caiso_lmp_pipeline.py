"""
CAISO LMP Data Pipeline
========================
Pulls Day-Ahead Hourly LMP data for NP15, SP15, ZP26 hubs via gridstatus,
chunks by month to avoid rate limits, handles timezones, decomposes components,
aligns to ERA5 hourly timestamps, and saves to Parquet.

Requirements:
    pip install gridstatus pandas pyarrow tqdm

Usage:
    python caiso_lmp_pipeline.py

Output:
    data/caiso_lmp_raw.parquet       — long-form, all components
    data/caiso_lmp_wide.parquet      — wide-form, ready for ERA5 alignment
    data/caiso_lmp_summary.csv       — quick stats for sanity check
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── gridstatus import (pip install gridstatus) ──────────────────────────────
try:
    import gridstatus
except ImportError:
    raise ImportError("Run: pip install gridstatus")

# ── CONFIG ──────────────────────────────────────────────────────────────────
START_DATE   = "2019-01-01"   # match your ERA5 hourly start
END_DATE     = "2024-12-31"   # match your ERA5 hourly end
MARKET       = "DAY_AHEAD_HOURLY"
LOCATION_TYPE = "HUB"         # NP15, SP15, ZP26
HUB_LOCATIONS = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]


# CAISO returns timestamps in UTC; we convert to US/Pacific for alignment
# with ERA5 data that you likely have in local time.
TARGET_TZ    = "US/Pacific"

# Retry settings for rate limits
MAX_RETRIES  = 3
RETRY_DELAY  = 30  # seconds between retries

OUTPUT_DIR   = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HELPERS ─────────────────────────────────────────────────────────────────

def month_ranges(start: str, end: str):
    """Generate (start, end) pairs for each calendar month in range."""
    current = pd.Timestamp(start)
    stop    = pd.Timestamp(end)
    while current <= stop:
        month_end = current + relativedelta(months=1) - pd.Timedelta(hours=1)
        yield current.strftime("%Y-%m-%d"), min(month_end, stop).strftime("%Y-%m-%d")
        current += relativedelta(months=1)


def fetch_month(caiso, start: str, end: str, retries: int = MAX_RETRIES) -> pd.DataFrame:
    """Fetch one month of CAISO DA hourly LMP with retry logic."""
    for attempt in range(retries):
        try:
            df = caiso.get_lmp(
                date=start,           
                end=end,
                market=MARKET,
                locations=HUB_LOCATIONS,   
                verbose=False,
            )
            return df
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [retry {attempt+1}/{retries}] {e}. Waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  [FAILED] {start} → {end}: {e}")
                return pd.DataFrame()


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names, timezones, and types.

    gridstatus CAISO returns columns:
        Time, Market, Location, LMP, Energy, Congestion, Loss
    Time is timezone-aware (UTC or US/Pacific depending on version).
    """
    if df.empty:
        return df

    df = df.copy()

    # ── Normalize column names ───────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Timezone: normalize everything to US/Pacific, then strip tz ─────────
    if "time" not in df.columns:
        raise ValueError(f"No 'time' column found. Got: {list(df.columns)}")

    if df["time"].dtype == object:
        df["time"] = pd.to_datetime(df["time"], utc=True)

    if df["time"].dt.tz is None:
        # Assume UTC if naive
        df["time"] = df["time"].dt.tz_localize("UTC")

    df["time"] = df["time"].dt.tz_convert(TARGET_TZ)
    # Keep tz-aware — helpful for DST-safe joins with ERA5
    # (ERA5 UTC timestamps can be converted at join time)

    # ── Keep only relevant columns ───────────────────────────────────────────
    keep = ["time", "location", "lmp", "energy", "congestion", "loss"]
    available = [c for c in keep if c in df.columns]
    df = df[available]

    # ── Numeric coercion ─────────────────────────────────────────────────────
    for col in ["lmp", "energy", "congestion", "loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Location standardisation ─────────────────────────────────────────────
    df["location"] = df["location"].str.upper().str.strip()

    return df


# ── MAIN PULL ────────────────────────────────────────────────────────────────

def pull_lmp() -> pd.DataFrame:
    caiso  = gridstatus.CAISO()
    ranges = list(month_ranges(START_DATE, END_DATE))

    chunks = []
    for start, end in tqdm(ranges, desc="Pulling CAISO LMP"):
        df = fetch_month(caiso, start, end)
        if not df.empty:
            df = clean_and_standardize(df)
            chunks.append(df)
        time.sleep(1)  # polite rate throttle

    if not chunks:
        raise RuntimeError("No data pulled — check credentials or date range.")

    full = pd.concat(chunks, ignore_index=True)
    full = full.sort_values(["time", "location"]).reset_index(drop=True)

    # Drop exact duplicates that sometimes appear at DST transitions
    full = full.drop_duplicates(subset=["time", "location"])

    return full


# ── WIDE FORM (for modelling) ────────────────────────────────────────────────

def make_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to one row per timestamp, columns like:
        LMP_NP15, Energy_NP15, Congestion_NP15, Loss_NP15,
        LMP_SP15, Energy_SP15, Congestion_SP15, Loss_SP15,
        LMP_ZP26, Energy_ZP26, Congestion_ZP26, Loss_ZP26

    Also adds derived features:
        Congestion_Spread  = Congestion_SP15 - Congestion_NP15  (Path 26 proxy)
        LMP_Spread         = LMP_SP15 - LMP_NP15
        hour_of_day, day_of_week, month, is_weekend
    """
    components = ["lmp", "energy", "congestion", "loss"]
    available  = [c for c in components if c in df.columns]

    wide = df.pivot_table(
        index="time",
        columns="location",
        values=available,
        aggfunc="mean",      # handles rare duplicates after dedup
    )

    # Flatten MultiIndex columns: (lmp, NP15) → LMP_NP15
    wide.columns = [f"{comp.capitalize()}_{loc}" for comp, loc in wide.columns]
    wide = wide.reset_index()

    # ── Derived spread features (key target for congestion modelling) ────────
    if "Congestion_SP15" in wide.columns and "Congestion_NP15" in wide.columns:
        wide["Congestion_Spread_SP15_NP15"] = wide["Congestion_SP15"] - wide["Congestion_NP15"]

    if "LMP_SP15" in wide.columns and "LMP_NP15" in wide.columns:
        wide["LMP_Spread_SP15_NP15"] = wide["LMP_SP15"] - wide["LMP_NP15"]

    # ── Calendar features ────────────────────────────────────────────────────
    t = wide["time"]
    wide["hour_of_day"]  = t.dt.hour
    wide["day_of_week"]  = t.dt.dayofweek        # 0=Mon, 6=Sun
    wide["month"]        = t.dt.month
    wide["year"]         = t.dt.year
    wide["is_weekend"]   = (t.dt.dayofweek >= 5).astype(int)
    wide["season"]       = t.dt.month.map(
        {12: "winter", 1: "winter", 2: "winter",
          3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer",
          9: "fall",  10: "fall",  11: "fall"}
    )

    return wide


# ── SUMMARY STATS ─────────────────────────────────────────────────────────────

def make_summary(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Quick stats table for sanity-checking the pull."""
    numeric_cols = df_wide.select_dtypes(include=[np.number]).columns.tolist()
    # Drop calendar features from stats
    exclude = ["hour_of_day", "day_of_week", "month", "year", "is_weekend"]
    stat_cols = [c for c in numeric_cols if c not in exclude]

    stats = df_wide[stat_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats["null_count"] = df_wide[stat_cols].isnull().sum()
    stats["null_pct"]   = (stats["null_count"] / len(df_wide) * 100).round(2)
    return stats


# ── ERA5 ALIGNMENT HELPER ─────────────────────────────────────────────────────

def align_with_era5(lmp_wide: pd.DataFrame, era5_path: str) -> pd.DataFrame:
    """
    Load ERA5 hourly data and join to LMP on timestamp.

    ERA5 data is in UTC. LMP is in US/Pacific (tz-aware).
    We convert ERA5 to US/Pacific before joining.

    ERA5 is expected as a Parquet/NetCDF with columns:
        time (UTC, or naive assumed UTC), lat, lon, variable_name...

    This function assumes you have already spatially averaged ERA5
    over CAISO zone bounding boxes into a single hourly series.
    If your ERA5 is still on a grid, see the docstring note below.

    Note — Spatial averaging ERA5 to CAISO zones:
        NP15 bounding box (approx): lat 36.5–42.0, lon -124.5 to -119.5
        SP15 bounding box (approx): lat 32.5–36.5, lon -120.5 to -114.5
        ZP26 bounding box (approx): lat 35.0–37.5, lon -121.0 to -118.0
        Use xarray to do: ds.sel(lat=slice(lo, hi), lon=slice(lo, hi)).mean(["lat","lon"])
    """
    era5 = pd.read_parquet(era5_path)

    # Coerce ERA5 time to UTC-aware, then Pacific
    if era5["time"].dt.tz is None:
        era5["time"] = era5["time"].dt.tz_localize("UTC")
    era5["time"] = era5["time"].dt.tz_convert(TARGET_TZ)

    merged = lmp_wide.merge(era5, on="time", how="inner")
    print(f"Merged dataset: {len(merged):,} rows "
          f"({merged['time'].min()} → {merged['time'].max()})")
    return merged


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Pulling CAISO DA Hourly LMP: {START_DATE} → {END_DATE}")
    print(f"Market: {MARKET} | Locations: NP15, SP15, ZP26 (HUB)")
    print("-" * 60)

    # 1. Pull raw data
    lmp_raw = pull_lmp()
    raw_path = os.path.join(OUTPUT_DIR, "caiso_lmp_raw.parquet")
    lmp_raw.to_parquet(raw_path, index=False)
    print(f"\n✓ Raw data saved: {raw_path}  ({len(lmp_raw):,} rows)")

    # 2. Make wide form
    lmp_wide = make_wide(lmp_raw)
    wide_path = os.path.join(OUTPUT_DIR, "caiso_lmp_wide.parquet")
    lmp_wide.to_parquet(wide_path, index=False)
    print(f"✓ Wide data saved: {wide_path}  ({len(lmp_wide):,} rows, {len(lmp_wide.columns)} cols)")

    # 3. Summary stats
    summary = make_summary(lmp_wide)
    summary_path = os.path.join(OUTPUT_DIR, "caiso_lmp_summary.csv")
    summary.to_csv(summary_path)
    print(f"✓ Summary saved:   {summary_path}")
    print("\nQuick stats (LMP + Congestion columns):")
    display_cols = [c for c in summary.index if "LMP" in c or "Congestion" in c or "Spread" in c]
    print(summary.loc[display_cols, ["mean", "std", "min", "50%", "max", "null_pct"]].to_string())

    # 4. Optional: align with ERA5 (uncomment when ERA5 parquet is ready)
    # ERA5_PATH = "path/to/your/era5_hourly_western_us.parquet"
    # if os.path.exists(ERA5_PATH):
    #     merged = align_with_era5(lmp_wide, ERA5_PATH)
    #     merged.to_parquet(os.path.join(OUTPUT_DIR, "caiso_lmp_era5_merged.parquet"), index=False)
    #     print(f"✓ Merged dataset saved.")

    print("\nDone. Next step: run caiso_lmp_baseline_analysis.py")
