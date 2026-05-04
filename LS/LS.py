import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from astropy.timeseries import LombScargle


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Lomb-Scargle period search for a variable-star light curve."
    )

    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_DIR, "clean_data.csv"),
        help="Input CSV file. Default: ../clean_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR,
        help="Directory for plots and result tables. Default: this LS folder.",
    )

    parser.add_argument("--time-col", default=None, help="Time column name, e.g. JD, HJD, MJD, BJD.")
    parser.add_argument("--signal-col", default=None, help="Signal column name, e.g. Mag, Vmag, Flux.")
    parser.add_argument("--error-col", default=None, help="Optional signal error column name.")

    parser.add_argument("--min-period", type=float, default=0.05, help="Minimum period in days.")
    parser.add_argument("--max-period", type=float, default=100.0, help="Maximum period in days.")
    parser.add_argument("--samples-per-peak", type=int, default=20)
    parser.add_argument("--n-best-peaks", type=int, default=5)
    parser.add_argument("--peak-separation-widths", type=float, default=5.0)

    parser.add_argument("--n-monte-carlo", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-window-widths", type=float, default=5.0)
    parser.add_argument("--mc-grid-size", type=int, default=5001)

    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=True,
        help="Show interactive matplotlib windows in addition to saving PNG files.",
    )
    parser.add_argument(
        "--no-show-plots",
        action="store_false",
        dest="show_plots",
        help="Only save PNG files without opening interactive plot windows.",
    )

    return parser.parse_args()


def normalize_name(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(columns, aliases, forbidden_aliases=None):
    normalized = {normalize_name(col): col for col in columns}
    alias_norms = [normalize_name(alias) for alias in aliases]
    forbidden_norms = [normalize_name(alias) for alias in (forbidden_aliases or [])]

    for alias in alias_norms:
        if alias in normalized:
            return normalized[alias]

    for alias in alias_norms:
        for col in columns:
            col_norm = normalize_name(col)
            if alias in col_norm and not any(bad in col_norm for bad in forbidden_norms):
                return col

    return None


def choose_columns(df, args):
    columns = list(df.columns)

    time_col = args.time_col or find_column(
        columns,
        ["jd", "hjd", "bjd", "mjd", "time", "date"],
        forbidden_aliases=["error", "err", "sigma", "uncertainty"],
    )
    signal_col = args.signal_col or find_column(
        columns,
        ["mag", "magnitude", "vmag", "gmag", "rmag", "imag", "flux", "brightness"],
        forbidden_aliases=["error", "err", "sigma", "uncertainty", "limit"],
    )

    if time_col is None:
        raise ValueError(f"No time column found. Available columns: {columns}")
    if signal_col is None:
        raise ValueError(f"No magnitude/flux column found. Available columns: {columns}")
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found. Available columns: {columns}")
    if signal_col not in df.columns:
        raise ValueError(f"Signal column '{signal_col}' not found. Available columns: {columns}")

    if args.error_col is not None:
        err_col = args.error_col
        if err_col not in df.columns:
            raise ValueError(f"Error column '{err_col}' not found. Available columns: {columns}")
    else:
        signal_norm = normalize_name(signal_col)
        if "flux" in signal_norm:
            error_aliases = ["flux error", "flux_err", "fluxerr", "e_flux", "flux sigma", "dy", "yerr"]
        elif "mag" in signal_norm:
            error_aliases = ["mag error", "mag_err", "magerr", "e_mag", "merr", "mag sigma", "dy", "yerr"]
        else:
            error_aliases = ["error", "err", "uncertainty", "sigma", "dy", "yerr"]

        err_col = find_column(columns, error_aliases)

    return time_col, signal_col, err_col


def robust_scatter(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scatter = 1.4826 * mad
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = np.std(values, ddof=1)
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = 1.0
    return scatter


def refine_peak_parabolic(freq_grid, power_grid, peak_idx):
    """Return a sub-grid peak frequency using a quadratic fit around one grid maximum."""
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


def is_magnitude_column(column_name):
    name = normalize_name(column_name)
    return "mag" in name or "magnitude" in name


args = parse_args()

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(args.input, comment="#")

if df.empty:
    raise ValueError("Input file is empty.")

# ============================================================
# CHOOSE COLUMNS
# ============================================================
time_col, signal_col, err_col = choose_columns(df, args)

# ============================================================
# EXTRACT ARRAYS
# ============================================================
t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
y = pd.to_numeric(df[signal_col], errors="coerce").to_numpy()

if err_col is not None:
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
    raise ValueError("min-period and max-period must be positive.")

if args.min_period >= args.max_period:
    raise ValueError("min-period must be smaller than max-period.")

if args.max_period > time_span:
    print(
        "WARNING: max-period is longer than the data time span. "
        "Long-period peaks may be poorly constrained."
    )

min_frequency = 1.0 / args.max_period
max_frequency = 1.0 / args.min_period

# Natural frequency resolution ~ 1 / T
freq_resolution = 1.0 / time_span

# ============================================================
# OUTPUT FILE NAMES
# ============================================================
input_path = os.path.abspath(args.input)
output_dir = os.path.abspath(args.output_dir)
base_name = os.path.splitext(os.path.basename(input_path))[0]
ls_tag = f"{base_name}_LS_MC{args.n_monte_carlo}_{args.min_period:.3f}-{args.max_period:.1f}d"

os.makedirs(output_dir, exist_ok=True)

OUTPUT_PLOT = os.path.join(output_dir, f"{ls_tag}_periodogram.png")
OUTPUT_PHASE = os.path.join(output_dir, f"{ls_tag}_phase_curve.png")
OUTPUT_PERIOD = os.path.join(output_dir, f"{ls_tag}_best_period.txt")
OUTPUT_TOP_PEAKS = os.path.join(output_dir, f"{ls_tag}_top_peaks.csv")
OUTPUT_MC_PERIODS = os.path.join(output_dir, f"{ls_tag}_mc_periods.csv")
OUTPUT_MC_HIST = os.path.join(output_dir, f"{ls_tag}_mc_period_hist.png")

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
    samples_per_peak=args.samples_per_peak,
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
local_peak_mask = np.zeros_like(power, dtype=bool)
local_peak_mask[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
candidate_peak_indices = np.where(local_peak_mask)[0]
candidate_peak_indices = candidate_peak_indices[np.argsort(power[candidate_peak_indices])[::-1]]

peak_indices = []
min_peak_separation = args.peak_separation_widths * freq_resolution

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
        "power": power[idx],
    })

pd.DataFrame(top_peaks_data).to_csv(OUTPUT_TOP_PEAKS, index=False)

# ============================================================
# MONTE CARLO: LOCAL SEARCH AROUND MAIN PEAK
# ============================================================
rng = np.random.default_rng(args.seed)
mc_periods = []

local_half_width = args.mc_window_widths * freq_resolution
mc_min_frequency = max(min_frequency, best_frequency - local_half_width)
mc_max_frequency = min(max_frequency, best_frequency + local_half_width)

if mc_min_frequency >= mc_max_frequency:
    raise ValueError("Monte Carlo local frequency window collapsed.")

if args.mc_grid_size < 5:
    raise ValueError("mc-grid-size must be at least 5.")

mc_frequency_grid = np.linspace(mc_min_frequency, mc_max_frequency, args.mc_grid_size)
if dy is not None:
    mc_noise_sigma = dy
    mc_noise_level = np.nanmedian(mc_noise_sigma)
    mc_noise_source = f"measured photometric errors from '{err_col}'"
else:
    best_model = ls.model(t, best_frequency)
    residuals = y_centered - best_model
    mc_noise_level = robust_scatter(residuals)
    mc_noise_sigma = np.full_like(y, mc_noise_level)
    mc_noise_source = "estimated from residual scatter around the best LS model"

print("=" * 70)
print("Running Lomb-Scargle + Monte Carlo")
print(f"Input file                : {input_path}")
print(f"Output directory          : {output_dir}")
print(f"Time column               : {time_col}")
print(f"Signal column             : {signal_col}")
print(f"Error column              : {err_col if err_col is not None else 'not found'}")
print(f"MC noise model            : {mc_noise_source}")
print(f"MC typical noise level    : {mc_noise_level:.10f}")
print(f"Points used               : {len(t)}")
print(f"Main LS best period       : {best_period:.10f} days")
print(f"Main LS best frequency    : {best_frequency:.10f} 1/day")
print(f"Time span                 : {time_span:.10f} days")
print(f"Frequency resolution ~    : {freq_resolution:.10f} 1/day")
print(f"Period range searched     : {args.min_period:.10f} .. {args.max_period:.10f} days")
print(f"MC local frequency range  : {mc_min_frequency:.10f} .. {mc_max_frequency:.10f} 1/day")
print(f"MC frequency grid size    : {args.mc_grid_size}")
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
    "period_days": mc_periods,
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
    f.write(f"Input file: {input_path}\n")
    f.write(f"Output directory: {output_dir}\n")
    f.write(f"Time column: {time_col}\n")
    f.write(f"Signal column: {signal_col}\n")
    f.write(f"Error column: {err_col if err_col is not None else 'none'}\n")
    f.write(f"Number of points: {len(t)}\n")
    f.write(f"Time span (days): {time_span:.10f}\n")
    f.write(f"Min period searched (days): {args.min_period}\n")
    f.write(f"Max period searched (days): {args.max_period}\n")
    f.write(f"Samples per peak: {args.samples_per_peak}\n\n")

    f.write(f"Best period (days): {best_period:.10f}\n")
    f.write(f"Best frequency (1/day): {best_frequency:.10f}\n")
    f.write(f"Best power: {best_power:.10f}\n\n")

    f.write("=== MONTE CARLO SETTINGS ===\n")
    f.write(f"Monte Carlo iterations: {args.n_monte_carlo}\n")
    f.write(f"Random seed: {args.seed}\n")
    f.write(f"MC local window half-width (1/day): {local_half_width:.10f}\n")
    f.write(f"MC frequency range (1/day): {mc_min_frequency:.10f} .. {mc_max_frequency:.10f}\n")
    f.write(f"MC frequency grid size: {args.mc_grid_size}\n")
    f.write("MC peak refinement: parabolic interpolation around the grid maximum\n")
    f.write(f"MC noise model: {mc_noise_source}\n")
    f.write(f"MC typical noise level: {mc_noise_level:.10f}\n")
    f.write("\n")

    f.write("=== MONTE CARLO RESULTS ===\n")
    f.write(f"Mean period (days): {period_mean_mc:.10f}\n")
    f.write(f"Median period (days): {period_median_mc:.10f}\n")
    f.write(f"Std period (days): {period_std_mc:.10f}\n")
    f.write(f"16th percentile (days): {period_p16:.10f}\n")
    f.write(f"84th percentile (days): {period_p84:.10f}\n\n")

    f.write(f"Final adopted result (symmetric): {best_period:.10f} +/- {period_error:.10f} days\n")
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
for file_path in [
    OUTPUT_PLOT,
    OUTPUT_PHASE,
    OUTPUT_PERIOD,
    OUTPUT_TOP_PEAKS,
    OUTPUT_MC_PERIODS,
    OUTPUT_MC_HIST,
]:
    print(f"- {os.path.relpath(file_path, PROJECT_DIR)}")
