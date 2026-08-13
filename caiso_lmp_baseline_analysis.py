"""
CAISO LMP Baseline Analysis
=============================
Runs the 7-step baseline analysis to validate the weather signal in LMP data
before committing to model architecture work.

Requires:
    data/caiso_lmp_wide.parquet   (from caiso_lmp_pipeline.py)

Optionally:
    data/caiso_lmp_era5_merged.parquet  (for weather correlation steps)

Produces plots in:
    figures/step1_distributions.png
    figures/step2_stl_decomp.png
    figures/step3_weather_correlations.png
    figures/step4_nonlinearity.png
    figures/step5_congestion_spread.png
    figures/step6_lag_structure.png
    figures/step7_regime_lmp.png

Usage:
    python caiso_lmp_baseline_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy import stats
from scipy.signal import correlate
from statsmodels.tsa.seasonal import STL

from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# Try scikit-learn for clustering (Step 7) — graceful skip if missing
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not found — Step 7 (regime clustering) will be skipped.")

WIDE_PATH   = "data/processed/caiso_lmp_wide_with_load.parquet"
MERGED_PATH = "data/aligned/caiso_lmp_era5_averaged.parquet"   # optional, for steps 3/4/5/6
FIG_DIR     = "baseline_analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)

STYLE = {
    "NP15":  "#2196F3",
    "SP15":  "#FF5722",
    "ZP26":  "#4CAF50",
    "spread": "#9C27B0",
}

# ── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(WIDE_PATH):
        raise FileNotFoundError(
            f"{WIDE_PATH} not found. Run caiso_lmp_pipeline.py first."
        )
    df = pd.read_parquet(WIDE_PATH)
    # Make sure time is datetime
    df["time"] = pd.to_datetime(df["time"])

    print(f"Loaded: {len(df):,} rows  |  {df['time'].min().date()} → {df['time'].max().date()}")

    era5 = None
    if os.path.exists(MERGED_PATH):
        era5 = pd.read_parquet(MERGED_PATH)
        era5["time"] = pd.to_datetime(era5["time"])
        print(f"ERA5 merged data also loaded: {len(era5):,} rows")

    return df, era5

# ────────────────────────────────────────────────────────────────────────────
# STEP 1 — Marginal Distribution Analysis
# ────────────────────────────────────────────────────────────────────────────

def step1_distributions(df):
    print("\nStep 1: Marginal distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Step 1 — LMP Component Distributions by Zone", fontsize=14, fontweight="bold")

    lmp_cols  = [c for c in df.columns if c.startswith("LMP_") and "Spread" not in c]
    cong_cols = [c for c in df.columns if c.startswith("Congestion_") and "Spread" not in c]

    for i, col in enumerate(lmp_cols[:3]):
        zone = col.split("_")[1]
        ax = axes[0, i]
        data = df[col].dropna()
        ax.hist(data, bins=100, color=STYLE.get(zone, "steelblue"), alpha=0.75, density=True)
        ax.axvline(data.median(), color="black", lw=1.5, linestyle="--", label=f"Median: ${data.median():.1f}")
        ax.set_title(f"LMP {zone}")
        ax.set_xlabel("$/MWh")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        # Clip x-axis at 99th percentile for readability
        ax.set_xlim(data.quantile(0.005), data.quantile(0.995))

    for i, col in enumerate(cong_cols[:3]):
        zone = col.split("_")[1]
        ax = axes[1, i]
        data = df[col].dropna()
        ax.hist(data, bins=100, color=STYLE.get(zone, "coral"), alpha=0.75, density=True)
        ax.axvline(0, color="black", lw=1, linestyle=":")
        ax.axvline(data.median(), color="darkred", lw=1.5, linestyle="--", label=f"Median: ${data.median():.1f}")
        ax.set_title(f"Congestion {zone}")
        ax.set_xlabel("$/MWh")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(data.quantile(0.005), data.quantile(0.995))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "step1_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    # Print key stats
    print("  Near-zero congestion fraction (|congestion| < $1/MWh):")
    for col in cong_cols:
        zone = col.split("_")[1]
        frac = (df[col].abs() < 1).mean()
        print(f"    {zone}: {frac:.1%}")

    print(f"  Saved: {path}")

# ────────────────────────────────────────────────────────────────────────────
# STEP 2 — STL Temporal Decomposition
# ────────────────────────────────────────────────────────────────────────────

def step2_stl_decomp(df):
    print("\nStep 2: STL decomposition...")
    col = "Lmp_SP15" if "Lmp_SP15" in df.columns else df.filter(like="Lmp_").columns[0]
    series = df.set_index("time")[col].dropna().sort_index()
    series.index = series.index.tz_localize(None)

    # STL with period=24 (daily) for hourly data
    stl = STL(series, period=24, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Step 2 — STL Decomposition: {col}", fontsize=14, fontweight="bold")

    labels = ["Observed", "Trend", "Seasonal (24h)", "Residual"]
    components = [series, result.trend, result.seasonal, result.resid]
    colors = ["steelblue", "darkorange", "green", "crimson"]

    for ax, label, comp, color in zip(axes, labels, components, colors):
        ax.plot(comp.index, comp.values, lw=0.6, color=color, alpha=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)

    # Annotate variance explained
    total_var  = np.var(series)
    trend_var  = np.var(result.trend.dropna())
    season_var = np.var(result.seasonal)
    resid_var  = np.var(result.resid)
    axes[0].set_title(
        f"Variance: Trend={trend_var/total_var:.1%}  "
        f"Seasonal={season_var/total_var:.1%}  "
        f"Residual={resid_var/total_var:.1%}",
        fontsize=9
    )

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "step2_stl_decomp.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Residual variance (weather target): {resid_var/total_var:.1%} of total")
    print(f"  Saved: {path}")

# ────────────────────────────────────────────────────────────────────────────
# STEP 3 — Weather Correlation Heatmap (requires ERA5 merged data)
# ────────────────────────────────────────────────────────────────────────────

ERA5_DISPLAY_NAMES = {
    "2m_temperature":                           "t2m",
    "2m_dewpoint_temperature":                  "d2m",
    "10m_u_component_of_wind":                  "u10",
    "10m_v_component_of_wind":                  "v10",
    "surface_pressure":                         "sp",
    "total_cloud_cover":                        "tcc",
    "total_column_water_vapour":                "tcwv",
    "surface_solar_radiation_downwards":        "ssrd",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
    "total_precipitation":                      "tp",
}

def step3_weather_correlations(era5):
    if era5 is None:
        print("\nStep 3: Skipped (no ERA5 merged data)")
        return

    print("\nStep 3: Weather correlation heatmap...")

    era5_cols = [c for c in era5.columns if c.split("_")[0] in ERA5_DISPLAY_NAMES.values()]
    lmp_targets = [c for c in era5.columns if
                   any(c.startswith(p) for p in ["Lmp_", "Congestion_", "Energy_"])
                   and "Spread" not in c]

    if not era5_cols or not lmp_targets:
        print("  Could not identify ERA5 or LMP columns in merged data.")
        return

    corr_matrix = pd.DataFrame(index=era5_cols, columns=lmp_targets, dtype=float)
    for e_col in era5_cols:
        for l_col in lmp_targets:
            valid = era5[[e_col, l_col]].dropna()
            if len(valid) > 100:
                corr_matrix.loc[e_col, l_col], _ = stats.spearmanr(valid[e_col], valid[l_col])

    corr_matrix.index = [ERA5_DISPLAY_NAMES.get(c, c) for c in corr_matrix.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(corr_matrix.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_xticks(range(len(lmp_targets)))
    ax.set_xticklabels(lmp_targets, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index, fontsize=9)
    ax.set_title("Step 3 — Spearman Correlations: ERA5 Variables × LMP Components", fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(corr_matrix.index)):
        for j in range(len(lmp_targets)):
            val = corr_matrix.values[i, j]
            if not np.isnan(float(val)):
                ax.text(j, i, f"{float(val):.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(float(val)) > 0.5 else "black")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "step3_weather_correlations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ────────────────────────────────────────────────────────────────────────────
# STEP 4 — Nonlinearity Check: Temperature × LMP
# ────────────────────────────────────────────────────────────────────────────

def step4_nonlinearity(era5):
    if era5 is None:
        print("\nStep 4: Skipped (no ERA5 merged data)")
        return

    temp_col = next((c for c in era5.columns if "t2m" in c), None)
    lmp_col  = "Lmp_SP15" if "Lmp_SP15" in era5.columns else next(
        (c for c in era5.columns if c.startswith("LMP_")), None)

    if not temp_col or not lmp_col:
        print("\nStep 4: Skipped (required columns not found)")
        return

    print("\nStep 4: Nonlinearity check (T_2m vs LMP)...")

    # Convert K → °C if ERA5 temperature in Kelvin
    temp = era5[temp_col].copy()
    if temp.median() > 200:
        temp = temp - 273.15

    data = pd.DataFrame({"temp": temp, "lmp": era5[lmp_col], "hour": era5["hour_of_day"]}).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Step 4 — Nonlinearity: {temp_col} vs {lmp_col}", fontsize=12, fontweight="bold")

    # Left: scatter colored by hour of day
    sc = axes[0].scatter(data["temp"], data["lmp"], c=data["hour"], cmap="twilight",
                         alpha=0.15, s=4, rasterized=True)
    plt.colorbar(sc, ax=axes[0], label="Hour of Day")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("LMP ($/MWh)")
    axes[0].set_ylim(data["lmp"].quantile(0.005), data["lmp"].quantile(0.995))
    axes[0].set_title("All Hours (color = hour of day)")
    axes[0].grid(alpha=0.3)

    # Right: median LMP by temp bin, faceted by peak/off-peak
    data["peak"] = data["hour"].between(16, 21).map({True: "Peak (16–21h)", False: "Off-Peak"})
    data["temp_bin"] = pd.cut(data["temp"], bins=20)
    grouped = data.groupby(["temp_bin", "peak"])["lmp"].median().reset_index()
    grouped["temp_mid"] = grouped["temp_bin"].apply(lambda x: x.mid)

    for peak_label, color in [("Peak (16–21h)", "crimson"), ("Off-Peak", "steelblue")]:
        sub = grouped[grouped["peak"] == peak_label]
        axes[1].plot(sub["temp_mid"], sub["lmp"], "o-", color=color, label=peak_label, lw=1.5, ms=4)

    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Median LMP ($/MWh)")
    axes[1].set_title("Median LMP by Temperature Bin & Peak Period")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "step4_nonlinearity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ────────────────────────────────────────────────────────────────────────────
# STEP 5 — Spatial Decomposition: Congestion Spread
# ────────────────────────────────────────────────────────────────────────────

def step5_congestion_spread(df, era5=None):
    print("\nStep 5: Congestion spread analysis (Path 26 proxy)...")
    spread_col = "Congestion_Spread_SP15_NP15"
    if spread_col not in df.columns:
        print("  Spread column not found — skipping.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle("Step 5 — SP15–NP15 Congestion Spread (Path 26 Proxy)", fontsize=12, fontweight="bold")

    # Top: Time series of spread (monthly mean)
    spread = df.set_index("time")[spread_col].resample("ME").mean()
    axes[0].fill_between(spread.index, spread.values, alpha=0.6, color=STYLE["spread"])
    axes[0].axhline(0, color="black", lw=1, linestyle="--")
    axes[0].set_ylabel("$/MWh (monthly mean)")
    axes[0].set_title("Monthly Mean Congestion Spread")
    axes[0].grid(alpha=0.3)

    # Bottom: Seasonal box plot
    df_tmp = df[["time", spread_col, "month"]].dropna()
    months = sorted(df_tmp["month"].unique())
    data_by_month = [df_tmp[df_tmp["month"] == m][spread_col].values for m in months]
    bp = axes[1].boxplot(data_by_month, positions=months, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(STYLE["spread"])
        patch.set_alpha(0.5)
    axes[1].axhline(0, color="black", lw=1, linestyle="--")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("$/MWh")
    axes[1].set_title("Seasonal Distribution of Congestion Spread")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
    axes[1].set_xticks(months)
    axes[1].set_xticklabels([month_labels[m-1] for m in months], rotation=0)
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "step5_congestion_spread.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ────────────────────────────────────────────────────────────────────────────
# STEP 6 — Lag Structure: Temperature → LMP Cross-Correlation
# ────────────────────────────────────────────────────────────────────────────

def step6_lag_structure(era5):
    if era5 is None:
        print("\nStep 6: Skipped (no ERA5 merged data)")
        return

    temp_col = next((c for c in era5.columns if "t2m" in c), None)
    lmp_col  = "Lmp_SP15" if "Lmp_SP15" in era5.columns else next(
        (c for c in era5.columns if c.startswith("LMP_")), None)

    if not temp_col or not lmp_col:
        print("\nStep 6: Skipped (required columns not found)")
        return

    print("\nStep 6: Lag structure (temperature → LMP cross-correlation)...")

    sub = era5[[temp_col, lmp_col]].dropna().head(8760 * 3)   # use 3 years max
    temp_z = (sub[temp_col] - sub[temp_col].mean()) / sub[temp_col].std()
    lmp_z  = (sub[lmp_col]  - sub[lmp_col].mean())  / sub[lmp_col].std()

    max_lag = 48
    xcorr = correlate(lmp_z.values, temp_z.values, mode="full")
    mid   = len(xcorr) // 2
    lags  = np.arange(-max_lag, max_lag + 1)
    xcorr_vals = xcorr[mid - max_lag: mid + max_lag + 1] / len(temp_z)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(lags, xcorr_vals, color="steelblue", alpha=0.7, width=0.8)
    ax.axvline(0, color="black", lw=1.5, linestyle="--")
    ax.axvline(lags[np.argmax(xcorr_vals)], color="crimson", lw=2,
               label=f"Peak lag: {lags[np.argmax(xcorr_vals)]}h")
    ax.set_xlabel("Lag (hours) — positive = temperature leads LMP")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(f"Step 6 — Cross-correlation: {temp_col} → {lmp_col}", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "step6_lag_structure.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Peak temperature-LMP lag: {lags[np.argmax(xcorr_vals)]} hours")
    print(f"  Saved: {path}")

# ────────────────────────────────────────────────────────────────────────────
# STEP 7 — Regime Identification: GMM over weather, LMP distributions by regime
# ────────────────────────────────────────────────────────────────────────────

def step7_regime_identification(era5):
    if era5 is None or not HAS_SKLEARN:
        print("\nStep 7: Skipped (no ERA5 data or sklearn not installed)")
        return

    temp_col = next((c for c in era5.columns if "t2m" in c), None)
    ssrd_col = next((c for c in era5.columns if "ssrd" in c), None)
    wind_u   = next((c for c in era5.columns if "u10" in c), None)
    wind_v   = next((c for c in era5.columns if "v10" in c), None)
    lmp_col  = "Lmp_SP15" if "Lmp_SP15" in era5.columns else next(
        (c for c in era5.columns if c.startswith("Lmp_")), None)

    feature_cols = [c for c in [temp_col, ssrd_col, wind_u, wind_v] if c]
    if len(feature_cols) < 2 or not lmp_col:
        print("\nStep 7: Skipped (insufficient columns)")
        return

    print("\nStep 7: Regime identification (GMM over weather features)...")

    sub = era5[feature_cols + [lmp_col]].dropna()
    X   = StandardScaler().fit_transform(sub[feature_cols])

    # Fit GMM with k=4 regimes — empirically reasonable for CA weather
    gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42, n_init=5)
    sub = sub.copy()
    sub["regime"] = gmm.fit_predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Step 7 — Weather Regime Identification (GMM, k=4)", fontsize=12, fontweight="bold")

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    regime_labels = []

    # Left: regime LMP box plots
    for r in range(4):
        regime_lmp = sub[sub["regime"] == r][lmp_col]
        regime_labels.append(f"R{r} (n={len(regime_lmp):,})")

    bp = axes[0].boxplot(
        [sub[sub["regime"] == r][lmp_col].values for r in range(4)],
        patch_artist=True, showfliers=False, positions=range(4)
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.6)
    axes[0].set_xticklabels(regime_labels, fontsize=9)
    axes[0].set_ylabel("LMP ($/MWh)")
    axes[0].set_title("LMP Distribution by Weather Regime")
    axes[0].grid(alpha=0.3, axis="y")

    # Right: regime mean weather profile (radar-style bar)
    regime_means = sub.groupby("regime")[feature_cols].mean()
    regime_means_norm = (regime_means - regime_means.min()) / (regime_means.max() - regime_means.min())
    x = np.arange(len(feature_cols))
    width = 0.2
    short_names = [ERA5_DISPLAY_NAMES.get(c, c[:8]) for c in feature_cols]
    for r in range(4):
        axes[1].bar(x + r * width, regime_means_norm.iloc[r], width,
                    label=f"R{r}", color=colors[r], alpha=0.7)
    axes[1].set_xticks(x + 1.5 * width)
    axes[1].set_xticklabels(short_names, fontsize=9)
    axes[1].set_ylabel("Normalized Mean Value")
    axes[1].set_title("Regime Weather Profiles (normalized)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "step7_regime_lmp.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    # ANOVA test: are regime LMP distributions significantly different?
    regime_groups = [sub[sub["regime"] == r][lmp_col].values for r in range(4)]
    f_stat, p_val = stats.f_oneway(*regime_groups)
    print(f"  ANOVA across regimes: F={f_stat:.1f}, p={p_val:.2e}")
    print(f"  (p << 0.05 = regimes have significantly different LMP distributions → ML is warranted)")
    print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# Step 8: Linear regression and HGBoost
# ---------------------------------------------------------------------------
def step8_baseline_models_dayahead(df, target="Congestion_spread"):
    """
    Weather-only day-ahead baselines. No LMP lag features.
    Target: next-day raw value of Congestion_SP15, Energy_SP15, or Lmp_SP15.
    At bid time (10am day D), we know ERA5 forecast for day D+1.

    Frame: features at time t → predict target[t+24].
    Persistence: predict target[t+24] ≈ target[t]  (same hour today = tomorrow).
    """
    print(f"\nDay-Ahead Weather-Only Baselines → {target}")

    df = df.copy()  # never mutate caller's DataFrame
    df["Congestion_spread"] = df["Congestion_SP15"] - df["Congestion_NP15"]

    # ── Weather lag features (24h, 48h, 168h back) ──────────────────────────────
    weather_cols = [
        c for c in df.columns
        if any(v in c for v in ["ssrd", "fdir", "t2m", "d2m", "u10", "v10",
                                  "wind_speed", "rh", "tcwv", "tp", "tcc"])
    ]
    lag_cols = []
    for lag in [24, 48, 168]:
        for col in weather_cols:
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df[col].shift(lag)
            lag_cols.append(lag_col)

    # ── Feature matrix: current weather + lags + load + calendar ────────────────
    calendar_cols = ["hour_of_day", "day_of_week", "month", "is_weekend"]
    load_cols     = [c for c in df.columns if "Load_SP15" in c]
    feature_cols  = [
        c for c in (weather_cols + lag_cols + load_cols + calendar_cols)
        if c in df.columns
    ]

    # ── Target: raw LMP 24 h ahead ───────────────────────────────────────────────
    # shift(-24): at row t, label = target[t+24]
    df["__y__"] = df[target].shift(-24)

    df_model = df[feature_cols + ["__y__"]].dropna()
    X = df_model[feature_cols].values
    y = df_model["__y__"].values

    print(f"  Samples: {len(X):,}  |  Features: {len(feature_cols)}")

    # ── Time-series cross-validation ─────────────────────────────────────────────
    # gap=24 prevents leakage: the 24-step forecast horizon overlaps the gap.
    tscv = TimeSeriesSplit(n_splits=5, gap=24, test_size=168)
    results = {}

    named_models = [
        ("Ridge",   Ridge(alpha=1.0)),
        ("HistGBT", HistGradientBoostingRegressor(
            max_iter=300, max_depth=5, learning_rate=0.05,
            l2_regularization=0.1, random_state=42,
        )),
    ]

    for name, model in named_models:
        maes, r2s = [], []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if name == "Ridge":
                # Fresh scaler per fold — no state leakage
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            maes.append(mean_absolute_error(y_te, y_pred))
            r2s.append(r2_score(y_te, y_pred))

        results[name] = {
            "MAE":     np.mean(maes),
            "MAE_std": np.std(maes),
            "R2":      np.mean(r2s),
        }
        print(f"  {name:10s}  MAE={results[name]['MAE']:.3f} ± "
              f"{results[name]['MAE_std']:.3f}  R²={results[name]['R2']:.3f}")

    # ── Persistence baseline ─────────────────────────────────────────────────────
    # Predict target[t+24] = target[t]  (same hour today → same hour tomorrow).
    # Evaluate on aligned pairs only.
    pers_actual = df[target].shift(-24)     # true value 24 h ahead
    pers_pred   = df[target]                # naive forecast: use current value
    valid        = pers_actual.notna() & pers_pred.notna()

    pers_mae = mean_absolute_error(pers_actual[valid], pers_pred[valid])
    pers_r2  = r2_score(pers_actual[valid], pers_pred[valid])
    results["Persist-24h"] = {"MAE": pers_mae, "MAE_std": 0.0, "R2": pers_r2}
    print(f"  {'Persist-24h':10s}  MAE={pers_mae:.3f}              R²={pers_r2:.3f}")
    print("  (Persistence: predict target[t+24] = target[t], same hour)")

    # ── Permutation importance on HistGBT ────────────────────────────────────────
    # Re-train on the first 80 % of the full aligned dataset (chronological split).
    split  = int(0.8 * len(X))
    hgbt   = dict(named_models)["HistGBT"]   # fresh reference, not the CV-trained one
    hgbt.fit(X[:split], y[:split])

    perm = permutation_importance(
        hgbt, X[split:], y[split:],
        n_repeats=10, random_state=42, n_jobs=-1,
    )
    perm_imp = (
        pd.Series(perm.importances_mean, index=feature_cols)
        .sort_values(ascending=False)
        .head(15)
    )
    print(f"\n  Top 15 permutation importances (HistGBT on held-out 20%):")
    print(perm_imp.round(4).to_string())

    return results

# ────────────────────────────────────────────────────────────────────────────
# Step 9: Anomaly detection
# ────────────────────────────────────────────────────────────────────────────

def step9_anomaly_detection_potential(df):
    # Does weather explain congestion WHEN it deviates from persistence?
    df["residual_24h"] = df["Congestion_SP15"] - df["Congestion_SP15"].shift(24)
    df["large_deviation"] = (df["residual_24h"].abs() > df["residual_24h"].abs().quantile(0.75)).astype(int)

    # Correlation of weather with the residual (deviation from persistence)
    weather_cols = [c for c in df.columns if any(v in c for v in ["ssrd", "fdir", "t2m", "wind_speed"])]
    residual_corr = df[weather_cols + ["residual_24h"]].corr()["residual_24h"].drop("residual_24h")
    print(residual_corr.sort_values())

    # And the classification signal
    from sklearn.metrics import roc_auc_score
    X_cls = df[weather_cols + ["hour_of_day", "month"]].dropna()
    y_cls = df.loc[X_cls.index, "large_deviation"]
    # Quick logistic regression AUC via HistGBT classifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    from sklearn.model_selection import cross_val_score
    auc = cross_val_score(clf, X_cls, y_cls, cv=TimeSeriesSplit(5), scoring="roc_auc")
    print(f"\nWeather → Large congestion deviation AUC: {auc.mean():.3f} ± {auc.std():.3f}")

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, era5 = load_data()

    #step1_distributions(df)
    #step2_stl_decomp(df)
    #step3_weather_correlations(era5)
    #step4_nonlinearity(era5)
    #step5_congestion_spread(df)
    #step6_lag_structure(era5)
    #step7_regime_identification(era5)
    step8_baseline_models_dayahead(era5)
    #step9_anomaly_detection_potential(era5)

    print("\n" + "=" * 60)
    print("Baseline analysis complete.")
    print(f"Figures saved in: {FIG_DIR}/")
    print("\nGo/No-go checklist:")
    print("  [ ] Step 1: Congestion near-zero >50% of hours? (expected: yes)")
    print("  [ ] Step 2: Residual variance >20% after STL? (check weather has room to explain)")
    print("  [ ] Step 3: |ρ| > 0.35 for T_2m and SSRD? (anti-pvlib trap check)")
    print("  [ ] Step 4: J-curve or U-curve in T vs LMP? (nonlinearity confirmed)")
    print("  [ ] Step 5: Spread spikes in summer (Aug-Sep)? (seasonal congestion signal)")
    print("  [ ] Step 6: Peak lag > 0h (temperature leads LMP)? (causal direction correct)")
    print("  [ ] Step 7: ANOVA p << 0.05? (regimes have distinct LMP distributions)")
    print("\nIf all 7 checks pass: proceed to iTransformer architecture.")
