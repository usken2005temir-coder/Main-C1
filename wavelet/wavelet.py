import argparse
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
MAX_PERIOD = 100.0
N_PERIODS = 240
N_TIME_CENTERS = 100

# Foster-style wavelet / WWZ parameter.
# Use 0.0 for automatic choice from the cadence and period range.
# Smaller values give wider time windows; larger values give narrower windows.
WAVELET_DECAY = 0.0

LS_LOCAL_WINDOW_FRACTION = 0.25

N_MONTE_CARLO = 1000
RANDOM_SEED = 42
MC_LOCAL_WINDOW_FRACTION = 0.10
MC_PERIODS = 121
MC_TIME_CENTERS = 25

MAP_SMOOTH_PERIOD_BINS = 3
MAP_SMOOTH_TIME_BINS = 5

SHOW_PLOTS = True

# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Foster-style wavelet analysis for variable-star light curves."
    )

    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)

    parser.add_argument("--time-col", default=None)
    parser.add_argument("--signal-col", default=None)
    parser.add_argument("--error-col", default=None)

    parser.add_argument("--min-period", type=float, default=MIN_PERIOD)
    parser.add_argument("--max-period", type=float, default=MAX_PERIOD)
    parser.add_argument("--n-periods", type=int, default=N_PERIODS)
    parser.add_argument("--n-time-centers", type=int, default=N_TIME_CENTERS)
    parser.add_argument(
        "--wavelet-decay",
        type=float,
        default=WAVELET_DECAY,
        help="Foster WWZ decay. Use 0 for automatic selection.",
    )

    parser.add_argument("--ls-period", type=float, default=None)
    parser.add_argument("--ls-report", default=None)
    parser.add_argument("--ls-local-window-fraction", type=float, default=LS_LOCAL_WINDOW_FRACTION)
    parser.add_argument(
        "--global-search",
        action="store_true",
        help="Ignore the LS period and search the full period range.",
    )

    parser.add_argument("--n-monte-carlo", type=int, default=N_MONTE_CARLO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--mc-window-fraction", type=float, default=MC_LOCAL_WINDOW_FRACTION)
    parser.add_argument("--mc-periods", type=int, default=MC_PERIODS)
    parser.add_argument("--mc-time-centers", type=int, default=MC_TIME_CENTERS)

    parser.add_argument("--map-smooth-periods", type=int, default=MAP_SMOOTH_PERIOD_BINS)
    parser.add_argument("--map-smooth-times", type=int, default=MAP_SMOOTH_TIME_BINS)

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
        ["Mag", "Magnitude", "Vmag", "gmag", "rmag", "imag", "Flux", "brightness"],
        forbidden=["error", "err", "sigma", "limit"]
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


def local_period_bounds(center_period, window_fraction, min_period, max_period):
    half_width = center_period * window_fraction
    local_min = max(min_period, center_period - half_width)
    local_max = min(max_period, center_period + half_width)

    if local_min >= local_max:
        raise ValueError("Local period window collapsed.")

    return local_min, local_max


def make_period_grid(min_period, max_period, n_periods):
    if n_periods < 5:
        raise ValueError("Period grid must contain at least 5 periods.")

    period_ratio = max_period / min_period

    if period_ratio >= 3.0:
        return np.geomspace(min_period, max_period, n_periods)

    return np.linspace(min_period, max_period, n_periods)


def make_time_grid(t, n_time_centers):
    if n_time_centers < 3:
        raise ValueError("Time grid must contain at least 3 time centers.")

    return np.linspace(t.min(), t.max(), n_time_centers)


def positive_time_step(t):
    diffs = np.diff(np.unique(t))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]

    if len(diffs) == 0:
        return 1.0

    return np.median(diffs)


def automatic_wavelet_decay(t, periods):
    time_span = t.max() - t.min()
    cadence = positive_time_step(t)

    # The time window must contain enough irregular observations.
    # For very short periods this intentionally makes the WWZ window broad.
    target_window = max(20.0 * cadence, 0.05 * time_span)
    target_period = np.median(periods)

    decay = (target_period / (2.0 * np.pi * target_window)) ** 2
    return max(decay, 1e-12)


def smooth_nan_matrix(matrix, period_window, time_window):
    result = np.array(matrix, dtype=float, copy=True)

    for axis, window in [(0, period_window), (1, time_window)]:
        window = int(window)

        if window <= 1:
            continue

        kernel = np.ones(window, dtype=float)
        valid = np.isfinite(result)
        values = np.where(valid, result, 0.0)

        numerator = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"),
            axis,
            values
        )
        denominator = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"),
            axis,
            valid.astype(float)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            result = numerator / denominator

        result[denominator <= 0] = np.nan

    return result

# ============================================================
# FOSTER-STYLE WEIGHTED WAVELET / WWZ-LIKE CORE
# ============================================================
def wavelet_matrix(t, y, periods, time_centers, decay):
    wwz = np.empty((len(periods), len(time_centers)), dtype=float)
    power = np.empty_like(wwz)
    n_eff = np.empty_like(wwz)

    for i, period in enumerate(periods):
        omega = 2.0 * np.pi / period
        x = omega * (t[np.newaxis, :] - time_centers[:, np.newaxis])

        weights = np.exp(-decay * x * x)
        weight_sum = np.sum(weights, axis=1)
        weight_square_sum = np.sum(weights * weights, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            n_eff_period = weight_sum * weight_sum / weight_square_sum

        cos_x = np.cos(x)
        sin_x = np.sin(x)

        wy = weights * y
        wy2 = weights * y * y

        s0 = weight_sum
        s1 = np.sum(weights * cos_x, axis=1)
        s2 = np.sum(weights * sin_x, axis=1)
        s11 = np.sum(weights * cos_x * cos_x, axis=1)
        s12 = np.sum(weights * cos_x * sin_x, axis=1)
        s22 = np.sum(weights * sin_x * sin_x, axis=1)

        b0 = np.sum(wy, axis=1)
        b1 = np.sum(wy * cos_x, axis=1)
        b2 = np.sum(wy * sin_x, axis=1)
        ywy = np.sum(wy2, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            total_var = ywy - (b0 * b0) / s0

        lhs = np.zeros((len(time_centers), 3, 3), dtype=float)
        lhs[:, 0, 0] = s0
        lhs[:, 0, 1] = s1
        lhs[:, 0, 2] = s2
        lhs[:, 1, 0] = s1
        lhs[:, 1, 1] = s11
        lhs[:, 1, 2] = s12
        lhs[:, 2, 0] = s2
        lhs[:, 2, 1] = s12
        lhs[:, 2, 2] = s22

        rhs = np.column_stack([b0, b1, b2])

        wwz_period = np.full(len(time_centers), np.nan)
        power_period = np.full(len(time_centers), np.nan)
        valid = np.isfinite(n_eff_period) & (n_eff_period > 3.0) & np.isfinite(total_var) & (total_var > 0)

        if np.any(valid):
            try:
                coeff = np.linalg.solve(lhs[valid], rhs[valid])
            except np.linalg.LinAlgError:
                coeff = np.array([
                    np.linalg.lstsq(lhs_row, rhs_row, rcond=None)[0]
                    for lhs_row, rhs_row in zip(lhs[valid], rhs[valid])
                ])

            beta_rhs = np.sum(coeff * rhs[valid], axis=1)
            residual_var = ywy[valid] - beta_rhs
            explained_var = total_var[valid] - residual_var

            good = np.isfinite(residual_var) & (residual_var > 0) & np.isfinite(explained_var)
            valid_indices = np.where(valid)[0]
            good_indices = valid_indices[good]

            if len(good_indices) > 0:
                explained_good = np.maximum(explained_var[good], 0.0)
                residual_good = residual_var[good]
                total_good = total_var[valid][good]
                n_eff_good = n_eff_period[valid][good]

                power_period[good_indices] = explained_good / total_good
                wwz_period[good_indices] = (n_eff_good - 3.0) * explained_good / (2.0 * residual_good)

        wwz[i, :] = wwz_period
        power[i, :] = power_period
        n_eff[i, :] = n_eff_period

    return wwz, power, n_eff


def mean_finite_by_period(matrix):
    finite_counts = np.sum(np.isfinite(matrix), axis=1)
    global_spectrum = np.full(matrix.shape[0], np.nan)
    valid = finite_counts > 0
    global_spectrum[valid] = np.nansum(matrix[valid], axis=1) / finite_counts[valid]

    if not np.any(np.isfinite(global_spectrum)):
        raise ValueError("Wavelet analysis failed: all global spectrum values are NaN.")

    return global_spectrum


def best_wavelet_period(periods, score_matrix):
    global_spectrum = mean_finite_by_period(score_matrix)

    best_idx = np.nanargmax(global_spectrum)
    return periods[best_idx], global_spectrum[best_idx], global_spectrum


def harmonic_phase_model(t, y, dy, period, n_points=500):
    phase = (t % period) / period
    angle = 2.0 * np.pi * phase

    design = np.column_stack([
        np.ones(len(t)),
        np.cos(angle),
        np.sin(angle),
    ])

    if dy is not None:
        weights = 1.0 / (dy * dy)
        weights = weights / np.nanmedian(weights)
    else:
        weights = np.ones(len(t))

    lhs = design.T @ (weights[:, np.newaxis] * design)
    rhs = design.T @ (weights * y)

    try:
        coeff = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    phase_grid = np.linspace(0.0, 2.0, n_points)
    angle_grid = 2.0 * np.pi * phase_grid
    model = coeff[0] + coeff[1] * np.cos(angle_grid) + coeff[2] * np.sin(angle_grid)

    return phase_grid, model

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
    raise ValueError("Too few valid data points for wavelet analysis.")

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

if args.wavelet_decay < 0:
    raise ValueError("wavelet-decay must be non-negative. Use 0 for automatic selection.")

if args.map_smooth_periods < 1 or args.map_smooth_times < 1:
    raise ValueError("Map smoothing values must be positive integers.")

y_centered = y - np.mean(y)

# ============================================================
# GET LS PERIOD AND BUILD SEARCH GRID
# ============================================================
ls_period, ls_source = read_ls_period(args)

if ls_period is not None and not args.global_search:
    if args.min_period <= ls_period <= args.max_period:
        period_min, period_max = local_period_bounds(
            ls_period,
            args.ls_local_window_fraction,
            args.min_period,
            args.max_period
        )
        search_mode = "local around LS period"
    else:
        period_min = args.min_period
        period_max = args.max_period
        search_mode = "global wavelet search; LS period outside requested range"
else:
    period_min = args.min_period
    period_max = args.max_period
    search_mode = "global wavelet search"

periods = make_period_grid(period_min, period_max, args.n_periods)
time_centers = make_time_grid(t, args.n_time_centers)
period_grid_mode = "logarithmic" if (period_max / period_min) >= 3.0 else "linear"

if args.wavelet_decay > 0:
    effective_wavelet_decay = args.wavelet_decay
    decay_mode = "manual"
else:
    effective_wavelet_decay = automatic_wavelet_decay(t, periods)
    decay_mode = "automatic"

# ============================================================
# OUTPUT FILE NAMES
# ============================================================
os.makedirs(args.output_dir, exist_ok=True)

base_name = os.path.splitext(os.path.basename(args.input))[0]
wavelet_tag = f"{base_name}_wavelet_MC{args.n_monte_carlo}_{period_min:.3f}-{period_max:.1f}d"

OUTPUT_MAP = os.path.join(args.output_dir, f"{wavelet_tag}_map.png")
OUTPUT_GLOBAL = os.path.join(args.output_dir, f"{wavelet_tag}_global_spectrum.png")
OUTPUT_PHASE = os.path.join(args.output_dir, f"{wavelet_tag}_phase_curve.png")
OUTPUT_PHASE_WAVELET = os.path.join(args.output_dir, f"{wavelet_tag}_phase_curve_wavelet.png")
OUTPUT_MC_HIST = os.path.join(args.output_dir, f"{wavelet_tag}_mc_period_hist.png")
OUTPUT_PERIOD = os.path.join(args.output_dir, f"{wavelet_tag}_best_period.txt")
OUTPUT_GLOBAL_CSV = os.path.join(args.output_dir, f"{wavelet_tag}_global_spectrum.csv")
OUTPUT_RIDGE_CSV = os.path.join(args.output_dir, f"{wavelet_tag}_ridge.csv")
OUTPUT_MC_PERIODS = os.path.join(args.output_dir, f"{wavelet_tag}_mc_periods.csv")
OUTPUT_MATRIX_CSV = os.path.join(args.output_dir, f"{wavelet_tag}_matrix.csv")

# ============================================================
# MAIN WAVELET ANALYSIS
# ============================================================
print("=" * 70)
print("Running Foster-style wavelet analysis")
print(f"Input file                : {args.input}")
print(f"Output directory          : {args.output_dir}")
print(f"Search mode               : {search_mode}")
print(f"LS period source          : {ls_source if ls_source is not None else 'not available'}")
print(f"LS period                 : {ls_period if ls_period is not None else 'not available'}")
print(f"Wavelet period range      : {period_min:.10f} .. {period_max:.10f} days")
print(f"Wavelet periods           : {len(periods)}")
print(f"Wavelet period grid       : {period_grid_mode}")
print(f"Wavelet time centers      : {len(time_centers)}")
print(f"Map smoothing             : {args.map_smooth_periods} x {args.map_smooth_times} bins (display only)")
print(f"Wavelet decay mode        : {decay_mode}")
print(f"Effective wavelet decay   : {effective_wavelet_decay:.12g}")
print(f"Points used               : {len(t)}")
print(f"Time span                 : {time_span:.10f} days")
print("=" * 70)

wwz, power, n_eff = wavelet_matrix(
    t,
    y_centered,
    periods,
    time_centers,
    effective_wavelet_decay
)

try:
    best_period, best_local_power, global_power_spectrum = best_wavelet_period(periods, power)
    global_wwz_spectrum = mean_finite_by_period(wwz)
except ValueError:
    target_window = max(time_span / 20.0, np.median(np.diff(t)) * 5.0)
    effective_wavelet_decay = (np.median(periods) / (2.0 * np.pi * target_window)) ** 2

    print("WARNING: Standard wavelet decay produced an empty map.")
    decay_mode = "adaptive retry"
    print(f"Retrying with adaptive wavelet decay: {effective_wavelet_decay:.12g}")

    wwz, power, n_eff = wavelet_matrix(
        t,
        y_centered,
        periods,
        time_centers,
        effective_wavelet_decay
    )

    best_period, best_local_power, global_power_spectrum = best_wavelet_period(periods, power)
    global_wwz_spectrum = mean_finite_by_period(wwz)

best_idx_for_report = int(np.nanargmin(np.abs(periods - best_period)))
best_wwz = global_wwz_spectrum[best_idx_for_report]

best_frequency = 1.0 / best_period

if ls_period is not None and period_min <= ls_period <= period_max:
    reference_period = ls_period
    reference_label = "LS reference period"
else:
    reference_period = best_period
    reference_label = "best wavelet period"

ridge_periods = np.full(len(time_centers), np.nan)
ridge_wwz = np.full(len(time_centers), np.nan)

for j in range(len(time_centers)):
    column = wwz[:, j]
    if np.any(np.isfinite(column)):
        ridge_idx = np.nanargmax(column)
        ridge_periods[j] = periods[ridge_idx]
        ridge_wwz[j] = column[ridge_idx]

# ============================================================
# MONTE CARLO AROUND THE REFERENCE PERIOD
# ============================================================
rng = np.random.default_rng(args.seed)
mc_periods = []

mc_min_period, mc_max_period = local_period_bounds(
    reference_period,
    args.mc_window_fraction,
    args.min_period,
    args.max_period
)

mc_period_grid = make_period_grid(mc_min_period, mc_max_period, args.mc_periods)
mc_time_grid = make_time_grid(t, args.mc_time_centers)

if dy is not None:
    mc_noise_sigma = dy
else:
    mc_noise_sigma = np.full_like(y, robust_scatter(y_centered))

print("=" * 70)
print("Running Wavelet Monte Carlo")
print(f"Monte Carlo iterations    : {args.n_monte_carlo}")
print(f"MC period range           : {mc_min_period:.10f} .. {mc_max_period:.10f} days")
print(f"MC period grid size       : {len(mc_period_grid)}")
print(f"MC time centers           : {len(mc_time_grid)}")
print("=" * 70)

for i in range(args.n_monte_carlo):
    y_mc = y + rng.normal(loc=0.0, scale=mc_noise_sigma, size=len(y))
    y_mc_centered = y_mc - np.mean(y_mc)

    _, power_mc, _ = wavelet_matrix(
        t,
        y_mc_centered,
        mc_period_grid,
        mc_time_grid,
        effective_wavelet_decay
    )

    best_period_mc, _, _ = best_wavelet_period(mc_period_grid, power_mc)
    mc_periods.append(best_period_mc)

    if (i + 1) % 100 == 0 or (i + 1) == args.n_monte_carlo:
        print(f"Monte Carlo: {i + 1}/{args.n_monte_carlo}")

mc_periods = np.array(mc_periods)

# ============================================================
# MONTE CARLO STATISTICS
# ============================================================
period_mean_mc = np.mean(mc_periods)
period_std_mc = np.std(mc_periods, ddof=1) if len(mc_periods) > 1 else 0.0
period_median_mc = np.median(mc_periods)
period_p16 = np.percentile(mc_periods, 16)
period_p84 = np.percentile(mc_periods, 84)

period_error = period_std_mc
period_err_minus = period_median_mc - period_p16
period_err_plus = period_p84 - period_median_mc

# ============================================================
# SAVE DATA TABLES
# ============================================================
pd.DataFrame({
    "period_days": periods,
    "frequency_per_day": 1.0 / periods,
    "mean_local_wavelet_power": global_power_spectrum,
    "mean_wwz_like_statistic": global_wwz_spectrum
}).to_csv(OUTPUT_GLOBAL_CSV, index=False)

pd.DataFrame({
    "time": time_centers,
    "time_minus_start_days": time_centers - t.min(),
    "ridge_period_days": ridge_periods,
    "ridge_frequency_per_day": 1.0 / ridge_periods,
    "ridge_wwz": ridge_wwz
}).to_csv(OUTPUT_RIDGE_CSV, index=False)

pd.DataFrame({
    "iteration": np.arange(1, len(mc_periods) + 1),
    "period_days": mc_periods
}).to_csv(OUTPUT_MC_PERIODS, index=False)

matrix_df = pd.DataFrame(wwz, columns=[f"time_minus_start_{value - t.min():.8f}" for value in time_centers])
matrix_df.insert(0, "period_days", periods)
matrix_df.to_csv(OUTPUT_MATRIX_CSV, index=False)

# ============================================================
# PRINT RESULTS
# ============================================================
print("\n" + "=" * 70)
print("FINAL WAVELET RESULTS")
print(f"Best wavelet period            : {best_period:.10f} days")
print(f"Best wavelet frequency         : {best_frequency:.10f} 1/day")
print(f"Best mean local wavelet power  : {best_local_power:.10f}")
print(f"WWZ-like statistic at best     : {best_wwz:.10f}")
print(f"Reference period for plots/MC  : {reference_period:.10f} days ({reference_label})")
if ls_period is not None:
    print(f"Difference from LS period      : {best_period - ls_period:.10f} days")
print("-" * 70)
print(f"Monte Carlo mean period        : {period_mean_mc:.10f} days")
print(f"Monte Carlo median period      : {period_median_mc:.10f} days")
print(f"Monte Carlo std                : {period_std_mc:.10f} days")
print(f"16th percentile                : {period_p16:.10f} days")
print(f"84th percentile                : {period_p84:.10f} days")
print("-" * 70)
print(f"FINAL RESULT (symmetric)       : P = {reference_period:.10f} +/- {period_error:.10f} days")
print(f"MC PERCENTILE RESULT           : P = {period_median_mc:.10f} (+{period_err_plus:.10f} / -{period_err_minus:.10f}) days")
print("=" * 70)

# ============================================================
# SAVE TEXT REPORT
# ============================================================
with open(OUTPUT_PERIOD, "w", encoding="utf-8") as f:
    f.write("=== WAVELET RESULTS ===\n")
    f.write("Method: Foster-style local weighted Fourier / WWZ-like wavelet diagnostic\n")
    f.write(f"Input file: {args.input}\n")
    f.write(f"Output directory: {args.output_dir}\n")
    f.write(f"Search mode: {search_mode}\n")
    f.write(f"LS period source: {ls_source if ls_source is not None else 'not available'}\n")
    f.write(f"LS period (days): {ls_period if ls_period is not None else 'not available'}\n")
    f.write(f"Time column: {time_col}\n")
    f.write(f"Signal column: {signal_col}\n")
    f.write(f"Error column: {err_col if err_col is not None else 'none'}\n")
    f.write(f"Number of points: {len(t)}\n")
    f.write(f"Time span (days): {time_span:.10f}\n")
    f.write(f"Min period searched (days): {period_min:.10f}\n")
    f.write(f"Max period searched (days): {period_max:.10f}\n")
    f.write(f"Period grid: {period_grid_mode}\n")
    f.write(f"Map smoothing for PNG only: {args.map_smooth_periods} x {args.map_smooth_times} bins\n")
    f.write(f"Wavelet decay mode: {decay_mode}\n")
    f.write(f"Requested wavelet decay: {args.wavelet_decay}\n")
    f.write(f"Effective wavelet decay: {effective_wavelet_decay}\n")
    f.write(f"Wavelet periods: {len(periods)}\n")
    f.write(f"Wavelet time centers: {len(time_centers)}\n\n")

    f.write(f"Best wavelet period (days): {best_period:.10f}\n")
    f.write(f"Best wavelet frequency (1/day): {best_frequency:.10f}\n")
    f.write(f"Best mean local wavelet power: {best_local_power:.10f}\n")
    f.write(f"WWZ-like statistic at best period: {best_wwz:.10f}\n")
    f.write(f"Reference period for plots/MC (days): {reference_period:.10f}\n")
    f.write(f"Reference period source: {reference_label}\n")
    if ls_period is not None:
        f.write(f"Difference from LS period (days): {best_period - ls_period:.10f}\n")
    f.write("\n")

    f.write("=== MONTE CARLO SETTINGS ===\n")
    f.write(f"Monte Carlo iterations: {args.n_monte_carlo}\n")
    f.write(f"Random seed: {args.seed}\n")
    f.write(f"MC period range (days): {mc_min_period:.10f} .. {mc_max_period:.10f}\n")
    f.write(f"MC period grid size: {len(mc_period_grid)}\n")
    f.write(f"MC time centers: {len(mc_time_grid)}\n")
    if err_col is None:
        f.write("MC noise model: no error column found; robust scatter was used\n")
    f.write("\n")

    f.write("=== MONTE CARLO RESULTS ===\n")
    f.write(f"Mean period (days): {period_mean_mc:.10f}\n")
    f.write(f"Median period (days): {period_median_mc:.10f}\n")
    f.write(f"Std period (days): {period_std_mc:.10f}\n")
    f.write(f"16th percentile (days): {period_p16:.10f}\n")
    f.write(f"84th percentile (days): {period_p84:.10f}\n\n")

    f.write(f"Final adopted result (symmetric): {reference_period:.10f} +/- {period_error:.10f} days\n")
    f.write(f"MC percentile result: {period_median_mc:.10f} (+{period_err_plus:.10f} / -{period_err_minus:.10f}) days\n")

# ============================================================
# PLOT: WAVELET MAP
# ============================================================
time_centers_plot = time_centers - t.min()
power_plot = smooth_nan_matrix(power, args.map_smooth_periods, args.map_smooth_times)

plt.figure(figsize=(12, 7))
finite_power = power_plot[np.isfinite(power_plot)]
if len(finite_power) > 0:
    color_min = np.nanpercentile(finite_power, 5)
    color_max = np.nanpercentile(finite_power, 99)
else:
    color_min = None
    color_max = None

mesh = plt.pcolormesh(
    time_centers_plot,
    periods,
    power_plot,
    shading="auto",
    cmap="magma",
    vmin=color_min,
    vmax=color_max
)
plt.colorbar(mesh, label="Local wavelet power")
plt.axhline(
    reference_period,
    color="white",
    linestyle=":",
    linewidth=1.8,
    label=f"Reference period = {reference_period:.6f} d"
)
plt.xlabel(f"Time - {t.min():.5f} (days)")
plt.ylabel("Period (days)")
plt.title("Wavelet diagnostic map: local power near the reference period")
if period_grid_mode == "logarithmic":
    plt.yscale("log")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUT_MAP, dpi=300)

if args.show_plots:
    plt.show()

plt.close()

# ============================================================
# PLOT: GLOBAL WAVELET SPECTRUM
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(periods, global_power_spectrum, linewidth=1.4)
plt.axvline(best_period, linestyle="--", label=f"Best wavelet = {best_period:.10f} d")
if ls_period is not None and period_min <= ls_period <= period_max:
    plt.axvline(ls_period, linestyle=":", label=f"LS period = {ls_period:.10f} d")
plt.xlabel("Period (days)")
plt.ylabel("Mean local wavelet power")
plt.title("Global wavelet power spectrum")
if period_grid_mode == "logarithmic":
    plt.xscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_GLOBAL, dpi=300)

if args.show_plots:
    plt.show()

plt.close()

# ============================================================
# PLOT: PHASE-FOLDED CURVE
# ============================================================
def save_phase_curve(period_value, label_text, output_path):
    phase = (t % period_value) / period_value
    phase_model, y_model = harmonic_phase_model(t, y, dy, period_value)

    plt.figure(figsize=(9, 6))
    plt.scatter(phase, y, s=18, alpha=0.8, label="Data")
    plt.scatter(phase + 1.0, y, s=18, alpha=0.8, label="Repeated phase")
    plt.plot(phase_model, y_model, color="black", linewidth=2.0, label="Weighted sine fit")
    plt.xlabel("Phase")
    plt.ylabel(signal_col)
    plt.title(f"Phase-folded curve ({label_text} = {period_value:.6f} d)")
    plt.grid(True, alpha=0.3)

    if is_magnitude_column(signal_col):
        plt.gca().invert_yaxis()

    plt.xlim(0, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    if args.show_plots:
        plt.show()

    plt.close()


save_phase_curve(reference_period, reference_label, OUTPUT_PHASE)

if abs(best_period - reference_period) > 1e-12:
    save_phase_curve(best_period, "best wavelet period", OUTPUT_PHASE_WAVELET)

# ============================================================
# PLOT: MONTE CARLO HISTOGRAM
# ============================================================
plt.figure(figsize=(10, 6))
plt.hist(mc_periods, bins=30, alpha=0.8)
plt.axvline(reference_period, linestyle="--", label=f"Reference = {reference_period:.10f} d")
if abs(best_period - reference_period) > 1e-12:
    plt.axvline(best_period, linestyle="-", alpha=0.7, label=f"Best wavelet = {best_period:.10f} d")
plt.axvline(period_mean_mc, linestyle="-.", label=f"MC mean = {period_mean_mc:.10f} d")
plt.axvline(period_p16, linestyle=":", label=f"P16 = {period_p16:.10f} d")
plt.axvline(period_p84, linestyle=":", label=f"P84 = {period_p84:.10f} d")
plt.xlabel("Period (days)")
plt.ylabel("Count")
plt.title("Monte Carlo distribution of wavelet periods")

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
print(f"- {os.path.relpath(OUTPUT_MAP, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_GLOBAL, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_PHASE, PROJECT_DIR)}")
if abs(best_period - reference_period) > 1e-12:
    print(f"- {os.path.relpath(OUTPUT_PHASE_WAVELET, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_MC_HIST, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_PERIOD, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_GLOBAL_CSV, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_RIDGE_CSV, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_MC_PERIODS, PROJECT_DIR)}")
print(f"- {os.path.relpath(OUTPUT_MATRIX_CSV, PROJECT_DIR)}")
