import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = "clean_data.csv"   # файл после sigma clipping

MIN_PERIOD = 0.05               # days
MAX_PERIOD = 100.0              # days
SAMPLES_PER_PEAK = 20
N_BEST_PEAKS = 5

# Monte Carlo
N_MONTE_CARLO = 1000
RANDOM_SEED = 42

# Local search around the main LS peak for MC
MC_LOCAL_WINDOW_IN_PEAK_WIDTHS = 5.0   # search window = +/- N * (1 / time_span)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_FILE, comment="#")

if df.empty:
    raise ValueError("Input file is empty.")

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================
required_cols = ["JD", "Mag"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {INPUT_FILE}. Available columns: {list(df.columns)}")

# We expect Mag Error to still exist after sigma clipping if it was present in APTEST/APTEST.csv
if "Mag Error" not in df.columns:
    raise ValueError(
        "Column 'Mag Error' was not found in clean_data.csv.\n"
        "Your Monte Carlo should use real errors, so make sure sigma clipping preserved this column."
    )

time_col = "JD"
signal_col = "Mag"
err_col = "Mag Error"

# ============================================================
# EXTRACT ARRAYS
# ============================================================
t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
y = pd.to_numeric(df[signal_col], errors="coerce").to_numpy()
dy = pd.to_numeric(df[err_col], errors="coerce").to_numpy()

# ============================================================
# CLEAN DATA
# ============================================================
mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)

t = t[mask]
y = y[mask]
dy = dy[mask]

if len(t) < 5:
    raise ValueError("Too few valid data points for Lomb-Scargle.")

# ============================================================
# SORT BY TIME
# ============================================================
order = np.argsort(t)
t = t[order]
y = y[order]
dy = dy[order]

# ============================================================
# BASIC INFO
# ============================================================
time_span = t.max() - t.min()
if time_span <= 0:
    raise ValueError("Time span must be positive.")

y_centered = y - np.mean(y)

# ============================================================
# PERIOD / FREQUENCY RANGE
# ============================================================
if MIN_PERIOD <= 0 or MAX_PERIOD <= 0:
    raise ValueError("MIN_PERIOD and MAX_PERIOD must be positive.")

if MIN_PERIOD >= MAX_PERIOD:
    raise ValueError("MIN_PERIOD must be smaller than MAX_PERIOD.")

min_frequency = 1.0 / MAX_PERIOD
max_frequency = 1.0 / MIN_PERIOD

# Natural frequency resolution ~ 1 / T
freq_resolution = 1.0 / time_span

# ============================================================
# OUTPUT FILE NAMES
# ============================================================
base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
ls_tag = f"{base_name}_LS_MC{N_MONTE_CARLO}_{MIN_PERIOD:.3f}-{MAX_PERIOD:.1f}d"

OUTPUT_PLOT = f"{ls_tag}_periodogram.png"
OUTPUT_PHASE = f"{ls_tag}_phase_curve.png"
OUTPUT_PERIOD = f"{ls_tag}_best_period.txt"
OUTPUT_TOP_PEAKS = f"{ls_tag}_top_peaks.csv"
OUTPUT_MC_PERIODS = f"{ls_tag}_mc_periods.csv"
OUTPUT_MC_HIST = f"{ls_tag}_mc_period_hist.png"

# ============================================================
# MAIN LOMB-SCARGLE
# ============================================================
ls = LombScargle(t, y_centered, dy=dy)
frequency, power = ls.autopower(
    minimum_frequency=min_frequency,
    maximum_frequency=max_frequency,
    samples_per_peak=SAMPLES_PER_PEAK
)

period = 1.0 / frequency

# sort for plotting in period space
sort_idx = np.argsort(period)
period_sorted = period[sort_idx]
power_sorted = power[sort_idx]

# ============================================================
# BEST PERIOD
# ============================================================
best_idx = np.argmax(power)
best_frequency = frequency[best_idx]
best_period = 1.0 / best_frequency
best_power = power[best_idx]

# ============================================================
# TOP PEAKS
# ============================================================
peak_indices = np.argsort(power)[::-1][:N_BEST_PEAKS]

top_peaks_data = []
for i, idx in enumerate(peak_indices, start=1):
    top_peaks_data.append({
        "rank": i,
        "period_days": 1.0 / frequency[idx],
        "frequency_per_day": frequency[idx],
        "power": power[idx]
    })

pd.DataFrame(top_peaks_data).to_csv(OUTPUT_TOP_PEAKS, index=False)

# ============================================================
# MONTE CARLO: LOCAL SEARCH AROUND MAIN PEAK
# ============================================================
rng = np.random.default_rng(RANDOM_SEED)
mc_periods = []

# local search window in frequency units
local_half_width = MC_LOCAL_WINDOW_IN_PEAK_WIDTHS * freq_resolution
mc_min_frequency = max(min_frequency, best_frequency - local_half_width)
mc_max_frequency = min(max_frequency, best_frequency + local_half_width)

if mc_min_frequency >= mc_max_frequency:
    raise ValueError("Monte Carlo local frequency window collapsed.")

print("=" * 70)
print("Running Lomb-Scargle + Monte Carlo")
print(f"Input file                : {INPUT_FILE}")
print(f"Points used               : {len(t)}")
print(f"Main LS best period       : {best_period:.10f} days")
print(f"Main LS best frequency    : {best_frequency:.10f} 1/day")
print(f"Time span                 : {time_span:.10f} days")
print(f"Frequency resolution ~    : {freq_resolution:.10f} 1/day")
print(f"MC local frequency range  : {mc_min_frequency:.10f} .. {mc_max_frequency:.10f} 1/day")
print(f"Monte Carlo iterations    : {N_MONTE_CARLO}")
print("=" * 70)

for i in range(N_MONTE_CARLO):
    # perturb each point according to its photometric uncertainty
    y_mc = y + rng.normal(loc=0.0, scale=dy, size=len(y))
    y_mc_centered = y_mc - np.mean(y_mc)

    ls_mc = LombScargle(t, y_mc_centered, dy=dy)
    freq_mc, power_mc = ls_mc.autopower(
        minimum_frequency=mc_min_frequency,
        maximum_frequency=mc_max_frequency,
        samples_per_peak=SAMPLES_PER_PEAK
    )

    best_idx_mc = np.argmax(power_mc)
    best_freq_mc = freq_mc[best_idx_mc]
    best_period_mc = 1.0 / best_freq_mc
    mc_periods.append(best_period_mc)

    if (i + 1) % 100 == 0:
        print(f"Monte Carlo: {i + 1}/{N_MONTE_CARLO}")

mc_periods = np.array(mc_periods)

# ============================================================
# MONTE CARLO STATISTICS
# ============================================================
period_mean_mc = np.mean(mc_periods)
period_std_mc = np.std(mc_periods, ddof=1)
period_median_mc = np.median(mc_periods)
period_p16 = np.percentile(mc_periods, 16)
period_p84 = np.percentile(mc_periods, 84)

# symmetric error
period_error = period_std_mc

# asymmetric errors
period_err_minus = best_period - period_p16
period_err_plus = period_p84 - best_period

# ============================================================
# SAVE MC PERIODS
# ============================================================
pd.DataFrame({
    "iteration": np.arange(1, len(mc_periods) + 1),
    "period_days": mc_periods
}).to_csv(OUTPUT_MC_PERIODS, index=False)

# ============================================================
# PRINT RESULTS
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print(f"Best LS period                 : {best_period:.10f} days")
print(f"Best LS frequency              : {best_frequency:.10f} 1/day")
print(f"Best LS power                  : {best_power:.10f}")
print("-" * 70)
print(f"Monte Carlo mean period        : {period_mean_mc:.10f} days")
print(f"Monte Carlo median period      : {period_median_mc:.10f} days")
print(f"Monte Carlo std                : {period_std_mc:.10f} days")
print(f"16th percentile                : {period_p16:.10f} days")
print(f"84th percentile                : {period_p84:.10f} days")
print("-" * 70)
print(f"FINAL RESULT (symmetric)       : P = {best_period:.10f} ± {period_error:.10f} days")
print(f"FINAL RESULT (asymmetric)      : P = {best_period:.10f} (+{period_err_plus:.10f} / -{period_err_minus:.10f}) days")
print("=" * 70)

print("\nTop peaks from the main LS search:")
for row in top_peaks_data:
    print(
        f"{row['rank']:2d}. "
        f"Period = {row['period_days']:.10f} days, "
        f"Frequency = {row['frequency_per_day']:.10f} 1/day, "
        f"Power = {row['power']:.10f}"
    )

# ============================================================
# SAVE TEXT REPORT
# ============================================================
with open(OUTPUT_PERIOD, "w", encoding="utf-8") as f:
    f.write("=== LOMB-SCARGLE RESULTS ===\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Time column: {time_col}\n")
    f.write(f"Signal column: {signal_col}\n")
    f.write(f"Error column: {err_col}\n")
    f.write(f"Number of points: {len(t)}\n")
    f.write(f"Time span (days): {time_span:.10f}\n")
    f.write(f"Min period searched (days): {MIN_PERIOD}\n")
    f.write(f"Max period searched (days): {MAX_PERIOD}\n")
    f.write(f"Samples per peak: {SAMPLES_PER_PEAK}\n\n")

    f.write(f"Best period (days): {best_period:.10f}\n")
    f.write(f"Best frequency (1/day): {best_frequency:.10f}\n")
    f.write(f"Best power: {best_power:.10f}\n\n")

    f.write("=== MONTE CARLO SETTINGS ===\n")
    f.write(f"Monte Carlo iterations: {N_MONTE_CARLO}\n")
    f.write(f"MC local window half-width (1/day): {local_half_width:.10f}\n")
    f.write(f"MC frequency range (1/day): {mc_min_frequency:.10f} .. {mc_max_frequency:.10f}\n\n")

    f.write("=== MONTE CARLO RESULTS ===\n")
    f.write(f"Mean period (days): {period_mean_mc:.10f}\n")
    f.write(f"Median period (days): {period_median_mc:.10f}\n")
    f.write(f"Std period (days): {period_std_mc:.10f}\n")
    f.write(f"16th percentile (days): {period_p16:.10f}\n")
    f.write(f"84th percentile (days): {period_p84:.10f}\n\n")

    f.write(f"Final adopted result (symmetric): {best_period:.10f} ± {period_error:.10f} days\n")
    f.write(f"Final adopted result (asymmetric): {best_period:.10f} (+{period_err_plus:.10f} / -{period_err_minus:.10f}) days\n")

# ============================================================
# PLOT: LS PERIODOGRAM
# ============================================================
plt.figure(figsize=(11, 6))
plt.plot(period_sorted, power_sorted, linewidth=1.2)
plt.axvline(best_period, linestyle="--", alpha=0.8, label=f"Best LS period = {best_period:.6f} d")
plt.xlabel("Period (days)")
plt.ylabel("Lomb-Scargle Power")
plt.title("Lomb-Scargle Periodogram")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.show()

# ============================================================
# PLOT: PHASE-FOLDED CURVE
# ============================================================
phase = (t % best_period) / best_period

plt.figure(figsize=(9, 6))
plt.scatter(phase, y, s=18, alpha=0.8, label="Data")
plt.scatter(phase + 1.0, y, s=18, alpha=0.8, label="Repeated phase")
plt.xlabel("Phase")
plt.ylabel("Magnitude")
plt.title(f"Phase-folded curve (LS period = {best_period:.6f} d)")
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()
plt.xlim(0, 2)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PHASE, dpi=300)
plt.show()

# ============================================================
# PLOT: MONTE CARLO HISTOGRAM
# ============================================================
plt.figure(figsize=(10, 6))
plt.hist(mc_periods, bins=30, alpha=0.8)
plt.axvline(best_period, linestyle="--", label=f"Best LS = {best_period:.6f} d")
plt.axvline(period_mean_mc, linestyle="-.", label=f"MC mean = {period_mean_mc:.6f} d")
plt.axvline(period_p16, linestyle=":", label=f"P16 = {period_p16:.6f} d")
plt.axvline(period_p84, linestyle=":", label=f"P84 = {period_p84:.6f} d")
plt.xlabel("Period (days)")
plt.ylabel("Count")
plt.title("Monte Carlo distribution of LS periods")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_MC_HIST, dpi=300)
plt.show()

# ============================================================
# FINAL MESSAGE
# ============================================================
print("\nSaved files:")
print(f"- {OUTPUT_PLOT}")
print(f"- {OUTPUT_PHASE}")
print(f"- {OUTPUT_PERIOD}")
print(f"- {OUTPUT_TOP_PEAKS}")
print(f"- {OUTPUT_MC_PERIODS}")
print(f"- {OUTPUT_MC_HIST}")
