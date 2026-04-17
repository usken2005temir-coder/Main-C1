import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# ============================================================
# SETTINGS
# ============================================================
FILENAME = "APTEST/APTEST.csv"   # <-- change to your file
MIN_PERIOD = 0.05                # days
MAX_PERIOD = 100.0               # days
N_PERIODS = 20000                # number of trial periods

# ============================================================
# LOAD FILE
# ============================================================
df = pd.read_csv(FILENAME, comment="#")

print("Loaded file:", FILENAME)
print("Columns found:", list(df.columns))

# ============================================================
# AUTO-DETECT REQUIRED COLUMNS
# ============================================================
def find_column(columns, names):
    for name in names:
        for col in columns:
            if name.lower() in col.lower():
                return col
    return None

time_col = find_column(df.columns, ["jd", "hjd", "mjd", "time"])
mag_col = find_column(df.columns, ["mag", "magnitude"])
err_col = find_column(df.columns, ["magerr", "mag_err", "error", "err", "uncert"])

if time_col is None:
    raise ValueError("Could not find time column. Expected something like JD / HJD / MJD / Time")

if mag_col is None:
    raise ValueError("Could not find magnitude column. Expected something like Mag / Magnitude")

print(f"Detected time column: {time_col}")
print(f"Detected magnitude column: {mag_col}")

if err_col is not None:
    print(f"Detected error column: {err_col}")
else:
    print("No error column detected. Lomb-Scargle will run without uncertainties.")

# ============================================================
# BUILD CLEAN DATAFRAME
# ============================================================
use_cols = [time_col, mag_col]
if err_col is not None and err_col not in use_cols:
    use_cols.append(err_col)

df_clean = df[use_cols].copy()

for col in use_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

df_clean = df_clean.dropna(subset=[time_col, mag_col]).copy()

if err_col is not None:
    df_clean = df_clean[np.isfinite(df_clean[err_col])]
    df_clean = df_clean[df_clean[err_col] > 0].copy()

df_clean = df_clean.sort_values(by=time_col).reset_index(drop=True)

rename_dict = {
    time_col: "JD",
    mag_col: "Mag"
}
if err_col is not None:
    rename_dict[err_col] = "MagErr"

df_clean = df_clean.rename(columns=rename_dict)

print(f"Cleaned rows: {len(df_clean)}")

if len(df_clean) < 5:
    raise ValueError("Not enough valid points after cleaning for Lomb-Scargle analysis.")

# ============================================================
# EXTRACT CLEAN ARRAYS
# ============================================================
t = df_clean["JD"].values
y = df_clean["Mag"].values
dy = df_clean["MagErr"].values if "MagErr" in df_clean.columns else None

y_mean = np.mean(y)
y_centered = y - y_mean

# ============================================================
# PERIOD RANGE
# ============================================================
if MIN_PERIOD <= 0 or MAX_PERIOD <= 0:
    raise ValueError("MIN_PERIOD and MAX_PERIOD must be positive.")

if MIN_PERIOD >= MAX_PERIOD:
    raise ValueError("MIN_PERIOD must be smaller than MAX_PERIOD.")

period_grid = np.linspace(MIN_PERIOD, MAX_PERIOD, N_PERIODS)
frequency_grid = 1.0 / period_grid

# ============================================================
# LOMB-SCARGLE IN PERIOD SPACE
# ============================================================
if dy is not None:
    ls = LombScargle(t, y_centered, dy=dy)
else:
    ls = LombScargle(t, y_centered)

power = ls.power(frequency_grid)

# ============================================================
# BEST PERIOD
# ============================================================
best_idx = np.argmax(power)
best_period = period_grid[best_idx]
best_frequency = frequency_grid[best_idx]
best_power = power[best_idx]

print("\n===== LOMB-SCARGLE RESULTS =====")
print(f"Best period:    {best_period:.8f} days")
print(f"Best frequency: {best_frequency:.8f} cycles/day")
print(f"Max power:      {best_power:.6f}")

# ============================================================
# FALSE ALARM PROBABILITY
# ============================================================
try:
    fap = ls.false_alarm_probability(best_power)
    print(f"False alarm probability: {fap:.6e}")
except Exception as e:
    fap = np.nan
    print("False alarm probability could not be computed:", e)

# ============================================================
# TOP PEAKS
# ============================================================
top_n = min(10, len(power))
top_idx = np.argsort(power)[-top_n:][::-1]

print("\nTop candidate periods:")
for i, idx in enumerate(top_idx, 1):
    print(
        f"{i:2d}. P = {period_grid[idx]:.8f} days   "
        f"f = {1.0 / period_grid[idx]:.8f} cycles/day   "
        f"power = {power[idx]:.6f}"
    )

# ============================================================
# SAVE CLEAN DATA
# ============================================================
df_clean.to_csv("cleaned_data_used_for_ls.csv", index=False)
print("Saved: cleaned_data_used_for_ls.csv")

# ============================================================
# SAVE PERIODOGRAM
# ============================================================
periodogram_df = pd.DataFrame({
    "period": period_grid,
    "frequency": frequency_grid,
    "power": power
})
periodogram_df.to_csv("lomb_scargle_periodogram.csv", index=False)
print("Saved: lomb_scargle_periodogram.csv")

# ============================================================
# SAVE BEST RESULT
# ============================================================
best_period_df = pd.DataFrame({
    "best_frequency": [best_frequency],
    "best_period": [best_period],
    "max_power": [best_power],
    "false_alarm_probability": [fap]
})
best_period_df.to_csv("best_period_ls.csv", index=False)
print("Saved: best_period_ls.csv")

# ============================================================
# PLOT PERIODOGRAM VS PERIOD
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(period_grid, power, linewidth=1.2)
plt.axvline(best_period, linestyle="--", linewidth=1.5,
            label=f"Best P = {best_period:.6f} d")
plt.xlabel("Period [days]")
plt.ylabel("Power")
plt.title("Lomb-Scargle Periodogram")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("lomb_scargle_periodogram_period.png", dpi=300)
plt.show()

# ============================================================
# PHASE FOLDING
# ============================================================
phase = (t / best_period) % 1.0

sort_idx = np.argsort(phase)
phase_sorted = phase[sort_idx]
mag_sorted = y[sort_idx]

if dy is not None:
    err_sorted = dy[sort_idx]

phase_fit = np.linspace(0, 1, 1000)
t_fit = phase_fit * best_period
model_mag = ls.model(t_fit, best_frequency) + y_mean

# ============================================================
# OPTIONAL ERROR CLIPPING FOR PLOT ONLY
# ============================================================
if dy is not None:
    finite_err = err_sorted[np.isfinite(err_sorted)]
    if len(finite_err) > 0:
        err_limit = np.percentile(finite_err, 95)
        err_plot = np.clip(err_sorted, None, err_limit)
    else:
        err_plot = err_sorted

# ============================================================
# PLOT PHASE-FOLDED LIGHT CURVE
# ============================================================
plt.figure(figsize=(12, 6))

if dy is not None:
    plt.errorbar(
        phase_sorted,
        mag_sorted,
        yerr=err_plot,
        fmt="o",
        markersize=3,
        capsize=1.5,
        elinewidth=0.7,
        alpha=0.75,
        label="Clean data"
    )
    plt.errorbar(
        phase_sorted + 1.0,
        mag_sorted,
        yerr=err_plot,
        fmt="o",
        markersize=3,
        capsize=1.5,
        elinewidth=0.7,
        alpha=0.45,
        label="Repeated phase"
    )
else:
    plt.scatter(
        phase_sorted,
        mag_sorted,
        s=12,
        alpha=0.75,
        label="Clean data"
    )
    plt.scatter(
        phase_sorted + 1.0,
        mag_sorted,
        s=12,
        alpha=0.45,
        label="Repeated phase"
    )

plt.plot(phase_fit, model_mag, linewidth=2.0, label="LS model")
plt.plot(phase_fit + 1.0, model_mag, linewidth=2.0)

plt.gca().invert_yaxis()
plt.ylim(20, 5)
plt.xlim(0, 2)

plt.xlabel("Phase")
plt.ylabel("Magnitude")
plt.title(f"Phase-folded Light Curve (P = {best_period:.6f} d)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("phase_folded_ls.png", dpi=300)
plt.show()

# ============================================================
# SAVE PHASE DATA
# ============================================================
phase_df = pd.DataFrame({
    "JD": t,
    "phase": phase,
    "Mag": y
}).sort_values("phase")

if dy is not None:
    phase_df["MagErr"] = dy

phase_df.to_csv("phase_folded_data.csv", index=False)
print("Saved: phase_folded_data.csv")

# ============================================================
# SAVE TOP PEAKS
# ============================================================
top_peaks_df = pd.DataFrame({
    "rank": np.arange(1, top_n + 1),
    "period": period_grid[top_idx],
    "frequency": frequency_grid[top_idx],
    "power": power[top_idx]
})
top_peaks_df.to_csv("top_ls_peaks.csv", index=False)
print("Saved: top_ls_peaks.csv")

# ============================================================
# FINAL MESSAGE
# ============================================================
print("\nSaved files:")
print(" - cleaned_data_used_for_ls.csv")
print(" - lomb_scargle_periodogram.csv")
print(" - best_period_ls.csv")
print(" - lomb_scargle_periodogram_period.png")
print(" - phase_folded_ls.png")
print(" - phase_folded_data.csv")
print(" - top_ls_peaks.csv")