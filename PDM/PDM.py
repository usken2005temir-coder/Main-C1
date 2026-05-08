import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================
# DEFAULT SETTINGS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
LS_DIR = os.path.join(PROJECT_DIR, "LS")

INPUT_FILE = os.path.join(PROJECT_DIR, "clean_data.csv")
OUTPUT_DIR = SCRIPT_DIR

MIN_PERIOD = 0.05
MAX_PERIOD = 300.0
N_PERIODS_GLOBAL = 10000
N_PERIODS_LOCAL = 5001

N_BINS = 10
MIN_POINTS_PER_BIN = 3

LS_LOCAL_WINDOW_FRACTION = 0.05

N_MONTE_CARLO = 1000
RANDOM_SEED = 42
MC_LOCAL_WINDOW_FRACTION = 0.02
MC_GRID_SIZE = 401

SHOW_PLOTS = True


def clean_old_outputs(output_dir, keep_paths, patterns):
    keep = {os.path.abspath(path) for path in keep_paths}
    removed = 0

    for pattern in patterns:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            path_abs = os.path.abspath(path)
            if path_abs in keep or not os.path.isfile(path_abs):
                continue

            os.remove(path_abs)
            removed += 1

    return removed


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal PDM analysis for variable-star light curves."
    )

    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)

    parser.add_argument("--time-col", default=None)
    parser.add_argument("--signal-col", default=None)
    parser.add_argument("--error-col", default=None)

    parser.add_argument("--min-period", type=float, default=MIN_PERIOD)
    parser.add_argument("--max-period", type=float, default=MAX_PERIOD)
    parser.add_argument("--n-periods-global", type=int, default=N_PERIODS_GLOBAL)
    parser.add_argument("--n-periods-local", type=int, default=N_PERIODS_LOCAL)

    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument("--min-points-per-bin", type=int, default=MIN_POINTS_PER_BIN)

    parser.add_argument("--ls-period", type=float, default=None)
    parser.add_argument("--ls-report", default=None)
    parser.add_argument("--ls-local-window-fraction", type=float, default=LS_LOCAL_WINDOW_FRACTION)

    parser.add_argument("--n-monte-carlo", type=int, default=N_MONTE_CARLO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--mc-window-fraction", type=float, default=MC_LOCAL_WINDOW_FRACTION)
    parser.add_argument("--mc-grid-size", type=int, default=MC_GRID_SIZE)

    parser.add_argument("--show-plots", action="store_true", default=SHOW_PLOTS)
    parser.add_argument("--no-show-plots", action="store_false", dest="show_plots")

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
        ["Mag", "Magnitude", "Vmag", "gmag", "rmag", "imag", "V/R", "VR", "Flux", "brightness", "V", "R", "I"],
        forbidden=["error", "err", "sigma", "limit", "lambda", "wave", "wavelength"]
    )

    if time_col is None:
        raise ValueError(f"No time column found. Available columns: {columns}")

    if signal_col is None:
        raise ValueError(f"No signal column found. Available columns: {columns}")

    if args.error_col is not None:
        err_col = args.error_col
        if err_col not in df.columns:
            raise ValueError(f"Error column '{err_col}' not found. Available columns: {columns}")
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
def read_input_table(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".csv":
        table = pd.read_csv(path, comment="#")
    elif extension == ".tsv":
        table = pd.read_csv(path, comment="#", sep="\t")
    else:
        table = pd.read_csv(path, comment="#", sep=r"\s+", engine="python")

    if len(table.columns) == 1:
        table = pd.read_csv(path, comment="#", sep=None, engine="python")

    if len(table.columns) == 1:
        table = pd.read_csv(path, comment="#", sep=r"\s+", engine="python")

    return table


def robust_scatter(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scatter = 1.4826 * mad

    if not np.isfinite(scatter) or scatter <= 0:
        scatter = np.std(values, ddof=1)

    if not np.isfinite(scatter) or scatter <= 0:
        scatter = 1.0

    return scatter


def phase_bin_residual_scatter(t, y, period, n_bins, min_points_per_bin):
    phase = (t % period) / period
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    model = np.full_like(y, np.nan, dtype=float)
    fallback_level = np.nanmedian(y)

    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (phase >= bin_edges[i]) & (phase <= bin_edges[i + 1])
        else:
            in_bin = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])

        if np.count_nonzero(in_bin) >= min_points_per_bin:
            model[in_bin] = np.nanmedian(y[in_bin])

    model = np.where(np.isfinite(model), model, fallback_level)
    residuals = y - model

    return robust_scatter(residuals[np.isfinite(residuals)])


def is_magnitude_column(column_name):
    name = normalize_name(column_name)
    return "mag" in name or "magnitude" in name


def display_label_for_signal(column_name):
    name = str(column_name)
    normalized = normalize_name(name)

    if "mag" in normalized or "magnitude" in normalized:
        return "Magnitude"

    if normalized == "vr":
        return "V/R"

    return name


def display_label_from_dataframe(df, signal_col):
    if "Signal Label" in df.columns:
        labels = df["Signal Label"].dropna().astype(str).unique()
        if len(labels) > 0 and labels[0].strip():
            return labels[0]

    return display_label_for_signal(signal_col)


def is_magnitude_label(label):
    name = normalize_name(label)
    return "mag" in name or "magnitude" in name


def latest_ls_report():
    if not os.path.isdir(LS_DIR):
        return None

    candidates = []
    for name in os.listdir(LS_DIR):
        if name.endswith("_best_period.txt"):
            candidates.append(os.path.join(LS_DIR, name))

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def read_ls_period(args):
    if args.ls_period is not None:
        return args.ls_period, "command line"

    report_path = args.ls_report or latest_ls_report()
    if report_path is None or not os.path.exists(report_path):
        return None, None

    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    patterns = [
        r"Best period \(days\):\s*([0-9.eE+-]+)",
        r"Best LS period\s*:\s*([0-9.eE+-]+)",
        r"Final adopted result.*?:\s*([0-9.eE+-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), report_path

    return None, report_path


def make_period_grid(min_period, max_period, n_periods):
    return np.linspace(min_period, max_period, n_periods)


def local_period_bounds(center_period, window_fraction, min_period, max_period):
    half_width = center_period * window_fraction
    local_min = max(min_period, center_period - half_width)
    local_max = min(max_period, center_period + half_width)

    if local_min >= local_max:
        raise ValueError("Local period window collapsed.")

    return local_min, local_max


def pdm_theta_grid(t, y, periods, n_bins, min_points_per_bin):
    total_var = np.var(y, ddof=1)
    if not np.isfinite(total_var) or total_var <= 0:
        return np.full(len(periods), np.nan)

    phase = (t[np.newaxis, :] / periods[:, np.newaxis]) % 1.0
    bin_index = np.floor(phase * n_bins).astype(np.int16)
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    theta_numerator = np.zeros(len(periods), dtype=float)
    valid_points = np.zeros(len(periods), dtype=int)

    for bin_id in range(n_bins):
        in_bin = bin_index == bin_id
        counts = np.sum(in_bin, axis=1)
        valid_bin = counts >= min_points_per_bin

        sums = in_bin @ y
        sums2 = in_bin @ (y * y)

        bin_numerator = np.zeros(len(periods), dtype=float)
        bin_numerator[valid_bin] = (
            sums2[valid_bin] - (sums[valid_bin] * sums[valid_bin]) / counts[valid_bin]
        )

        theta_numerator += bin_numerator
        valid_points += counts * valid_bin

    theta = np.full(len(periods), np.nan)
    valid_theta = valid_points > n_bins
    theta[valid_theta] = theta_numerator[valid_theta] / ((valid_points[valid_theta] - 1) * total_var)

    return theta


def refine_pdm_minimum(period_grid, theta_grid, min_idx):
    if min_idx <= 0 or min_idx >= len(period_grid) - 1:
        return period_grid[min_idx]

    x0 = period_grid[min_idx]
    x = period_grid[min_idx - 1:min_idx + 2] - x0
    y_theta = theta_grid[min_idx - 1:min_idx + 2]

    if not np.all(np.isfinite(y_theta)):
        return period_grid[min_idx]

    a, b, _ = np.polyfit(x, y_theta, 2)

    if a <= 0:
        return period_grid[min_idx]

    refined_period = x0 - b / (2.0 * a)

    if period_grid[min_idx - 1] <= refined_period <= period_grid[min_idx + 1]:
        return refined_period

    return period_grid[min_idx]


def best_pdm_period(t, y, periods, n_bins, min_points_per_bin):
    theta = pdm_theta_grid(t, y, periods, n_bins, min_points_per_bin)

    if not np.any(np.isfinite(theta)):
        raise ValueError("PDM failed: all theta values are NaN.")

    best_idx = np.nanargmin(theta)
    best_period = refine_pdm_minimum(periods, theta, best_idx)
    best_theta = theta[best_idx]

    return best_period, best_theta, theta

# ============================================================
# LOAD FILE
# ============================================================
args = parse_args()

df = read_input_table(args.input)

if df.empty:
    raise ValueError("Input file is empty.")

# ============================================================
# FIND COLUMNS
# ============================================================
time_col, signal_col, err_col = choose_columns(df, args)
signal_label = display_label_from_dataframe(df, signal_col)

print("=" * 70)
print("Detected columns")
print(f"Time column   : {time_col}")
print(f"Signal column : {signal_col}")
print(f"Signal label  : {signal_label}")
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
    raise ValueError("Too few valid data points for PDM.")

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

if args.min_period <= 0 or args.max_period <= 0:
    raise ValueError("MIN_PERIOD and MAX_PERIOD must be positive.")

if args.min_period >= args.max_period:
    raise ValueError("MIN_PERIOD must be smaller than MAX_PERIOD.")

if args.n_bins < 2:
    raise ValueError("n-bins must be at least 2.")

if args.mc_grid_size < 5:
    raise ValueError("mc-grid-size must be at least 5.")

y_centered = y - np.mean(y)

# ============================================================
# GET LS PERIOD OR FALL BACK TO GLOBAL PDM SEARCH
# ============================================================
ls_period, ls_source = read_ls_period(args)

if ls_period is not None and args.min_period <= ls_period <= args.max_period:
    search_mode = "local around LS period"
    local_min_period, local_max_period = local_period_bounds(
        ls_period,
        args.ls_local_window_fraction,
        args.min_period,
        args.max_period
    )
    period_grid = make_period_grid(local_min_period, local_max_period, args.n_periods_local)
else:
    search_mode = "global PDM search"
    if ls_period is not None:
        print("WARNING: LS period was found but lies outside the requested PDM period range.")
    local_min_period = args.min_period
    local_max_period = args.max_period
    period_grid = make_period_grid(args.min_period, args.max_period, args.n_periods_global)

# ============================================================
# OUTPUT FILE NAMES
# ============================================================
os.makedirs(args.output_dir, exist_ok=True)

base_name = os.path.splitext(os.path.basename(args.input))[0]
pdm_tag = f"{base_name}_PDM_MC{args.n_monte_carlo}_{args.min_period:.3f}-{args.max_period:.1f}d"

OUTPUT_THETA = os.path.join(args.output_dir, f"{pdm_tag}_theta.png")
OUTPUT_PHASE = os.path.join(args.output_dir, f"{pdm_tag}_phase_curve.png")
OUTPUT_PERIOD = os.path.join(args.output_dir, f"{pdm_tag}_best_period.txt")
OUTPUT_TOP_MINIMA = os.path.join(args.output_dir, f"{pdm_tag}_top_minima.csv")
OUTPUT_MC_PERIODS = os.path.join(args.output_dir, f"{pdm_tag}_mc_periods.csv")
OUTPUT_MC_HIST = os.path.join(args.output_dir, f"{pdm_tag}_mc_period_hist.png")

# ============================================================
# MAIN PDM SEARCH
# ============================================================
best_period, best_theta, theta = best_pdm_period(
    t,
    y_centered,
    period_grid,
    args.n_bins,
    args.min_points_per_bin
)

# ============================================================
# TOP PDM MINIMA
# ============================================================
local_min_mask = np.zeros_like(theta, dtype=bool)
local_min_mask[1:-1] = (theta[1:-1] < theta[:-2]) & (theta[1:-1] < theta[2:])
candidate_indices = np.where(local_min_mask & np.isfinite(theta))[0]
candidate_indices = candidate_indices[np.argsort(theta[candidate_indices])]

top_minima_data = []
min_period_separation = best_period * 0.01

for idx in candidate_indices:
    candidate_period = period_grid[idx]

    if all(abs(candidate_period - row["period_days"]) >= min_period_separation for row in top_minima_data):
        top_minima_data.append({
            "rank": len(top_minima_data) + 1,
            "period_days": candidate_period,
            "frequency_per_day": 1.0 / candidate_period,
            "theta": theta[idx]
        })

    if len(top_minima_data) == 5:
        break

pd.DataFrame(top_minima_data).to_csv(OUTPUT_TOP_MINIMA, index=False)

# ============================================================
# MONTE CARLO AROUND THE PDM PERIOD
# ============================================================
rng = np.random.default_rng(args.seed)
mc_periods = []

mc_min_period, mc_max_period = local_period_bounds(
    best_period,
    args.mc_window_fraction,
    args.min_period,
    args.max_period
)
mc_period_grid = make_period_grid(mc_min_period, mc_max_period, args.mc_grid_size)

if dy is not None:
    mc_noise_sigma = dy
    mc_noise_level = np.nanmedian(mc_noise_sigma)
    mc_noise_source = f"measured photometric errors from '{err_col}'"
else:
    mc_noise_level = phase_bin_residual_scatter(
        t,
        y_centered,
        best_period,
        args.n_bins,
        args.min_points_per_bin
    )
    mc_noise_sigma = np.full_like(y, mc_noise_level)
    mc_noise_source = "estimated from residual scatter around the best PDM phase-bin model"

print("=" * 70)
print("Running PDM + Monte Carlo")
print(f"Input file                : {args.input}")
print(f"Output directory          : {args.output_dir}")
print(f"Search mode               : {search_mode}")
print(f"LS period source          : {ls_source if ls_source is not None else 'not available'}")
print(f"LS period                 : {ls_period if ls_period is not None else 'not available'}")
print(f"PDM search range          : {local_min_period:.10f} .. {local_max_period:.10f} days")
print(f"Time column               : {time_col}")
print(f"Signal column             : {signal_col}")
print(f"Signal label              : {signal_label}")
print(f"Error column              : {err_col if err_col is not None else 'not found'}")
print(f"MC noise model            : {mc_noise_source}")
print(f"MC typical noise level    : {mc_noise_level:.10f}")
print(f"Points used               : {len(t)}")
print(f"Time span                 : {time_span:.10f} days")
print(f"PDM best period           : {best_period:.10f} days")
print(f"PDM best theta            : {best_theta:.10f}")
print(f"Monte Carlo iterations    : {args.n_monte_carlo}")
print(f"MC period range           : {mc_min_period:.10f} .. {mc_max_period:.10f} days")
print("=" * 70)

for i in range(args.n_monte_carlo):
    y_mc = y + rng.normal(loc=0.0, scale=mc_noise_sigma, size=len(y))
    y_mc_centered = y_mc - np.mean(y_mc)

    best_period_mc, _, _ = best_pdm_period(
        t,
        y_mc_centered,
        mc_period_grid,
        args.n_bins,
        args.min_points_per_bin
    )
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
print("FINAL PDM RESULTS")
print(f"Best PDM period                : {best_period:.10f} days")
print(f"Best PDM frequency             : {1.0 / best_period:.10f} 1/day")
print(f"Best PDM theta                 : {best_theta:.10f}")
if ls_period is not None:
    print(f"Difference from LS period      : {best_period - ls_period:.10f} days")
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
    f.write("=== PDM RESULTS ===\n")
    f.write(f"Input file: {args.input}\n")
    f.write(f"Output directory: {args.output_dir}\n")
    f.write(f"Search mode: {search_mode}\n")
    f.write(f"LS period source: {ls_source if ls_source is not None else 'not available'}\n")
    f.write(f"LS period (days): {ls_period if ls_period is not None else 'not available'}\n")
    f.write(f"Time column: {time_col}\n")
    f.write(f"Signal column: {signal_col}\n")
    f.write(f"Signal label: {signal_label}\n")
    f.write(f"Error column: {err_col if err_col is not None else 'none'}\n")
    f.write(f"Number of points: {len(t)}\n")
    f.write(f"Time span (days): {time_span:.10f}\n")
    f.write(f"Min period searched (days): {args.min_period}\n")
    f.write(f"Max period searched (days): {args.max_period}\n")
    f.write(f"PDM bins: {args.n_bins}\n")
    f.write(f"Minimum points per bin: {args.min_points_per_bin}\n\n")

    f.write(f"Best PDM period (days): {best_period:.10f}\n")
    f.write(f"Best PDM frequency (1/day): {1.0 / best_period:.10f}\n")
    f.write(f"Best PDM theta: {best_theta:.10f}\n")
    if ls_period is not None:
        f.write(f"Difference from LS period (days): {best_period - ls_period:.10f}\n")
    f.write("\n")

    f.write("=== MONTE CARLO SETTINGS ===\n")
    f.write(f"Monte Carlo iterations: {args.n_monte_carlo}\n")
    f.write(f"Random seed: {args.seed}\n")
    f.write(f"MC period range (days): {mc_min_period:.10f} .. {mc_max_period:.10f}\n")
    f.write(f"MC grid size: {args.mc_grid_size}\n")
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
# PLOT: PDM THETA CURVE
# ============================================================
plt.figure(figsize=(11, 6))
plt.plot(period_grid, theta, linewidth=1.2)
plt.axvline(best_period, linestyle="--", alpha=0.8, label=f"Best PDM period = {best_period:.6f} d")
if ls_period is not None:
    plt.axvline(ls_period, linestyle=":", alpha=0.8, label=f"LS period = {ls_period:.6f} d")
plt.xlabel("Period (days)")
plt.ylabel("PDM theta")
plt.title("PDM theta curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_THETA, dpi=300)

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
plt.ylabel(signal_label)
plt.title(f"Phase-folded curve (PDM period = {best_period:.6f} d)")
plt.grid(True, alpha=0.3)

if is_magnitude_label(signal_label):
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
plt.axvline(best_period, linestyle="--", label=f"Best PDM = {best_period:.10f} d")
plt.axvline(period_mean_mc, linestyle="-.", label=f"MC mean = {period_mean_mc:.10f} d")
plt.axvline(period_p16, linestyle=":", label=f"P16 = {period_p16:.10f} d")
plt.axvline(period_p84, linestyle=":", label=f"P84 = {period_p84:.10f} d")
plt.xlabel("Period (days)")
plt.ylabel("Count")
plt.title("Monte Carlo distribution of PDM periods")

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
current_outputs = [
    OUTPUT_THETA,
    OUTPUT_PHASE,
    OUTPUT_PERIOD,
    OUTPUT_TOP_MINIMA,
    OUTPUT_MC_PERIODS,
    OUTPUT_MC_HIST,
]

removed_old_outputs = clean_old_outputs(
    args.output_dir,
    current_outputs,
    [
        "*_PDM_MC*_theta.png",
        "*_PDM_MC*_phase_curve.png",
        "*_PDM_MC*_best_period.txt",
        "*_PDM_MC*_top_minima.csv",
        "*_PDM_MC*_mc_periods.csv",
        "*_PDM_MC*_mc_period_hist.png",
    ],
)

if removed_old_outputs > 0:
    print(f"\nRemoved old PDM output files: {removed_old_outputs}")

print("\nSaved files:")
for file_path in current_outputs:
    print(f"- {os.path.relpath(file_path, PROJECT_DIR)}")
