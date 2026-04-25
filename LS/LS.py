import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from astropy.timeseries import LombScargle

# ============================================================
# DEFAULT SETTINGS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(PROJECT_DIR, "clean_data.csv")
OUTPUT_DIR = SCRIPT_DIR

MIN_PERIOD = 0.05
MAX_PERIOD = 100.0
SAMPLES_PER_PEAK = 20
N_BEST_PEAKS = 5
MIN_PEAK_SEPARATION_IN_WIDTHS = 5.0

N_MONTE_CARLO = 1000
RANDOM_SEED = 42
MC_LOCAL_WINDOW_IN_PEAK_WIDTHS = 5.0
MC_GRID_SIZE = 5001

SHOW_PLOTS = True

# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Lomb-Scargle analysis for variable-star light curves."
    )

    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)

    parser.add_argument("--time-col", default=None)
    parser.add_argument("--signal-col", default=None)
    parser.add_argument("--error-col", default=None)

    parser.add_argument("--min-period", type=float, default=MIN_PERIOD)
    parser.add_argument("--max-period", type=float, default=MAX_PERIOD)
    parser.add_argument("--samples-per-peak", type=int, default=SAMPLES_PER_PEAK)
    parser.add_argument("--n-best-peaks", type=int, default=N_BEST_PEAKS)

    parser.add_argument("--n-monte-carlo", type=int, default=N_MONTE_CARLO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--mc-grid-size", type=int, default=MC_GRID_SIZE)

    parser.add_argument("--show-plots", action="store_true", default=SHOW_PLOTS)

    return parser.parse_args()

# ============================================================
# COLUMN DETECTION
# ============================================================
def normalize_name(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(columns, names, forbidden=None):
    forbidden = forbidden or []

    for name in names:
        name_norm = normalize_name(name)

        for col in columns:
            col_norm = normalize_name(col)

            if name_norm == col_norm:
                return col

        for col in columns:
            col_norm = normalize_name(col)

            if name_norm in col_norm:
                if not any(normalize_name(bad) in col_norm for bad in forbidden):
                    return col

    return None


def choose_columns(df, args):
    columns = list(df.columns)

    time_col = args.time_col or find_column(
        columns,
        ["JD", "HJD", "BJD", "MJD", "time", "date"],
        forbidden=["error", "err", "sigma"]
    )

    signal_col = args.signal_col or find_column(
        columns,
        ["Mag", "Magnitude", "Vmag", "gmag", "rmag", "imag", "Flux", "brightness"],
        forbidden=["error", "err", "sigma", "limit"]
    )

    if time_col is None:
        raise ValueError(f"No time column found. Available columns: {columns}")

    if signal_col is None:
        raise ValueError(f"No signal column found. Available columns: {columns}")

    if args.error_col is not None:
        err_col = args.error_col
    else:
        signal_name = normalize_name(signal_col)

        if "flux" in signal_name:
            err_col = find_column(
                columns,
                ["Flux Error", "flux_err", "fluxerr", "e_flux", "dy", "yerr"]
            )
        else:
            err_col = find_column(
                columns,
                ["Mag Error", "mag_err", "magerr", "e_mag", "merr", "dy", "yerr"]
            )

    return time_col, signal_col, err_col

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def robust_scatter(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scatter = 1.4826 * mad

    if not np.isfinite(scatter) or scatter <= 0:
        scatter = np.std(values, ddof=1)

    if not np.isfinite(scatter) or scatter <= 0:
        scatter = 1.0

    return scatter


def is_magnitude_column(column_name):
    name = normalize_name(column_name)
    return "mag" in name or "magnitude" in name


def refine_peak_parabolic(freq_grid, power_grid, peak_idx):
    if peak_idx <= 0 or peak_idx >= len(freq_grid) - 1:
        return freq_grid[peak_idx]

    x0 = freq_grid[peak_idx]
    x = freq_grid[peak_idx - 1:peak_idx + 2] - x0
    y_power = power_grid[peak_idx - 1:peak_idx + 2]

    a, b, _ = np.polyfit(x, y_power, 2)

    if a >= 0:
        return freq_grid[peak_idx]

    refined_frequency = x0 - b / (2.0 * a)

    if freq_grid[peak_idx - 1] <= refined_frequency <= freq_grid[peak_idx + 1]:
        return refined_frequency

    return freq_grid[peak_idx]

# ============================================================
# LOAD FILE
# ============================================================
args = parse_args()

df = pd.read_csv(args.input, comment="#")

if df.empty:
    raise ValueError("Input file is empty.")

# ============================================================
# FIND COLUMNS
# ============================================================
time_col, signal_col, err_col = choose_columns(df, args)

print("=" * 70)
print("Detected columns")
print(f"Time column   : {time_col}")
print(f"Signal column : {signal_col}")
print(f"Error column  : {err_col if err_col is not None else 'not found'}")
print("=" * 70)

# ============================================================
# EXTRACT ARRAYS
# ============================================================
t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
y = pd.to_numeric(df[signal_col], errors="coerce").to_numpy()

if err_col is not None and err_col in df.columns:
    dy = pd.to_numeric(df[err_col], errors="coerce").to_numpy()
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
else:
    dy = None
    mask = np.isfinite(t) & np.isfinite(y)

t = t[mask]
y = y[mask]

if dy is not None:
    dy = dy[mask]

if len(t) < 5:
    raise ValueError("Too few valid data points for Lomb-Scargle.")

# ============================================================
# SORT BY TIME
# ============================================================
order = np.argsort(t)
t = t[order]
y = y[order]

if dy is not None:
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
if args.min_period <= 0 or args.max_period <= 0:
    raise ValueError("MIN_PERIOD and MAX_PERIOD must be positive.")

if args.min_period >= args.max_period:
    raise ValueError("MIN_PERIOD must be smaller than MAX_PERIOD.")

min_frequency = 1.0 / args.max_period
max_frequency = 1.0 / args.min_period

freq_resolution = 1.0 / time_span

# ============================================================
# OUTPUT FILE NAMES
# ============================================================
os.makedirs(args.output_dir, exist_ok=True)

base_name = os.path.splitext(os.path.basename(args.input))[0]
ls_tag = f"{base_name}_LS_MC{args.n_monte_carlo}_{args.min_period:.3f}-{args.max_period:.1f}d"

OUTPUT_PLOT = os.path.join(args.output_dir, f"{ls_tag}_periodogram.png")
OUTPUT_PHASE = os.path.join(args.output_dir, f"{ls_tag}_phase_curve.png")
OUTPUT_PERIOD = os.path.join(args.output_dir, f"{ls_tag}_best_period.txt")
OUTPUT_TOP_PEAKS = os.path.join(args.output_dir, f"{ls_tag}_top_peaks.csv")
OUTPUT_MC_PERIODS = os.path.join(args.output_dir, f"{ls_tag}_mc_periods.csv")
OUTPUT_MC_HIST = os.path.join(args.output_dir, f"{ls_tag}_mc_period_hist.png")

# ============================================================
# MAIN LOMB-SCARGLE
# ============================================================
if dy is not None:
    ls = LombScargle(t, y_centered, dy=dy)
else:
    ls = LombScargle(t, y_centered)

frequency, power = ls.autopower(
    minimum_frequency=min_frequency,
    maximum_frequency=max_frequency,
    samples_per_peak=args.samples_per_peak
)

period = 1.0 / frequency

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
local_peak_mask = np.zeros_like(power, dtype=bool)
local_peak_mask[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])

candidate_peak_indices = np.where(local_peak_mask)[0]
candidate_peak_indices = candidate_peak_indices[np.argsort(power[candidate_peak_indices])[::-1]]

peak_indices = []
min_peak_separation = MIN_PEAK_SEPARATION_IN_WIDTHS * freq_resolution

for idx in candidate_peak_indices:
    if all(abs(frequency[idx] - frequency[old_idx]) >= min_peak_separation for old_idx in peak_indices):
        peak_indices.append(idx)

    if len(peak_indices) == args.n_best_peaks:
        break

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
# MONTE CARLO
# ============================================================
rng = np.random.default_rng(args.seed)
mc_periods = []

local_half_width = MC_LOCAL_WINDOW_IN_PEAK_WIDTHS * freq_resolution
mc_min_frequency = max(min_frequency, best_frequency - local_half_width)
mc_max_frequency = min(max_frequency, best_frequency + local_half_width)

if mc_min_frequency >= mc_max_frequency:
    raise ValueError("Monte Carlo local frequency window collapsed.")

mc_frequency_grid = np.linspace(mc_min_frequency, mc_max_frequency, args.mc_grid_size)

if dy is not None:
    mc_noise_sigma = dy
else:
    mc_noise_sigma = np.full_like(y, robust_scatter(y_centered))

print("=" * 70)
print("Running Lomb-Scargle + Monte Carlo")
print(f"Input file                : {args.input}")
print(f"Output directory          : {args.output_dir}")
print(f"Points used               : {len(t)}")
print(f"Main LS best period       : {best_period:.10f} days")
print(f"Main LS best frequency    : {best_frequency:.10f} 1/day")
print(f"Time span                 : {time_span:.10f} days")
print(f"Frequency resolution ~    : {freq_resolution:.10f} 1/day")
print(f"Period range searched     : {args.min_period} .. {args.max_period} days")
print(f"Monte Carlo iterations    : {args.n_monte_carlo}")
print("=" * 70)

for i in range(args.n_monte_carlo):
    y_mc = y + rng.normal(loc=0.0, scale=mc_noise_sigma, size=len(y))
    y_mc_centered = y_mc - np.mean(y_mc)

    if dy is not None:
        ls_mc = LombScargle(t, y_mc_centered, dy=dy)
    else:
        ls_mc = LombScargle(t, y_mc_centered)

    power_mc = ls_mc.power(mc_frequency_grid)

    best_idx_mc = np.argmax(power_mc)
    best_freq_mc = refine_peak_parabolic(mc_frequency_grid, power_mc, best_idx_mc)
    best_period_mc = 1.0 / best_freq_mc

    mc_periods.append(best_period_mc)

    if (i + 1) % 100 == 0 or (i + 1) == args.n_monte_carlo:
        print(f"Monte Carlo: {i + 1}/{args.n_monte_carlo}")

mc_periods = np.array(mc_periods)

# ============================================================
# MONTE CARLO STATISTICS
# ============================================================
period_mean_mc = np.mean(mc_periods)
period_std_mc = np.std(mc_periods, ddof=1)
period_median_mc = np.median(mc_periods)
period_p16 = np.percentile(mc_periods, 16)
period_p84 = np.percentile(mc_periods, 84)

period_error = period_std_mc
period_err_minus = period_median_mc - period_p16
period_err_plus = period_p84 - period_median_mc

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
print(f"FINAL RESULT (symmetric)       : P = {best_period:.10f} +/- {period_error:.10f} days")
print(f"MC PERCENTILE RESULT           : P = {period_median_mc:.10f} (+{period_err_plus:.10f} / -{period_err_minus:.10f}) days")
print("=" * 70)

# ============================================================
# SAVE TEXT REPORT
# ============================================================
with open(OUTPUT_PERIOD, "w", encoding="utf-8") as f:
    f.write("=== LOMB-SCARGLE RESULTS ===\n")
    f.write(f"Input file: {args.input}\n")
    f.write(f"Time column: {time_col}\n")
    f.write(f"Signal column: {signal_col}\n")
    f.write(f"Error column: {err_col if err_col is not None else 'none'}\n")
    f.write(f"Number of points: {len(t)}\n")
    f.write(f"Time span (days): {time_span:.10f}\n")
    f.write(f"Min period searched (days): {args.min_period}\n")
    f.write(f"Max period searched (days): {args.max_period}\n\n")

    f.write(f"Best period (days): {best_period:.10f}\n")
    f.write(f"Best frequency (1/day): {best_frequency:.10f}\n")
    f.write(f"Best power: {best_power:.10f}\n\n")

    f.write("=== MONTE CARLO RESULTS ===\n")
    f.write(f"Mean period (days): {period_mean_mc:.10f}\n")
    f.write(f"Median period (days): {period_median_mc:.10f}\n")
    f.write(f"Std period (days): {period_std_mc:.10f}\n")
    f.write(f"16th percentile (days): {period_p16:.10f}\n")
    f.write(f"84th percentile (days): {period_p84:.10f}\n\n")

    f.write(f"Final adopted result: {best_period:.10f} +/- {period_error:.10f} days\n")
    f.write(f"MC percentile result: {period_median_mc:.10f} (+{period_err_plus:.10f} / -{period_err_minus:.10f}) days\n")

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

if args.show_plots:
    plt.show()

plt.close()

# ============================================================
# PLOT: PHASE-FOLDED CURVE
# ============================================================
phase = (t % best_period) / best_period

plt.figure(figsize=(9, 6))
plt.scatter(phase, y, s=18, alpha=0.8, label="Data")
plt.scatter(phase + 1.0, y, s=18, alpha=0.8, label="Repeated phase")
plt.xlabel("Phase")
plt.ylabel(signal_col)
plt.title(f"Phase-folded curve (LS period = {best_period:.6f} d)")
plt.grid(True, alpha=0.3)

if is_magnitude_column(signal_col):
    plt.gca().invert_yaxis()

plt.xlim(0, 2)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PHASE, dpi=300)

if args.show_plots:
    plt.show()

plt.close()

# ============================================================
# PLOT: MONTE CARLO HISTOGRAM
# ============================================================
plt.figure(figsize=(10, 6))
plt.hist(mc_periods, bins=30, alpha=0.8)
plt.axvline(best_period, linestyle="--", label=f"Best LS = {best_period:.10f} d")
plt.axvline(period_mean_mc, linestyle="-.", label=f"MC mean = {period_mean_mc:.10f} d")
plt.axvline(period_p16, linestyle=":", label=f"P16 = {period_p16:.10f} d")
plt.axvline(period_p84, linestyle=":", label=f"P84 = {period_p84:.10f} d")
plt.xlabel("Period (days)")
plt.ylabel("Count")
plt.title("Monte Carlo distribution of LS periods")

ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax.ticklabel_format(axis="x", style="plain", useOffset=False)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_MC_HIST, dpi=300)

if args.show_plots:
    plt.show()

plt.close()

# ============================================================
# FINAL MESSAGE
# ============================================================
print("\nSaved files:")
print(f"- {os.path.relpath(OUTPUT_PLOT, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_PHASE, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_PERIOD, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_TOP_PEAKS, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_MC_PERIODS, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_MC_HIST, PROJECT_DIR)}")
