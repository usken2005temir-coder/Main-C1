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
MAX_PERIOD = 100.0

N_FREQUENCIES_GLOBAL = 20000
N_FREQUENCIES_LOCAL = 10001
FOURIER_CHUNK_SIZE = 512

N_BEST_PEAKS = 5
MIN_PEAK_SEPARATION_IN_WIDTHS = 5.0
LS_LOCAL_WINDOW_FRACTION = 0.05

N_MONTE_CARLO = 1000
RANDOM_SEED = 42
MC_LOCAL_WINDOW_FRACTION = 0.02
MC_GRID_SIZE = 501

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
        description="Universal weighted Fourier analysis for variable-star light curves."
    )

    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)

    parser.add_argument("--time-col", default=None)
    parser.add_argument("--signal-col", default=None)
    parser.add_argument("--error-col", default=None)

    parser.add_argument("--min-period", type=float, default=MIN_PERIOD)
    parser.add_argument("--max-period", type=float, default=MAX_PERIOD)
    parser.add_argument("--n-frequencies-global", type=int, default=N_FREQUENCIES_GLOBAL)
    parser.add_argument("--n-frequencies-local", type=int, default=N_FREQUENCIES_LOCAL)
    parser.add_argument("--chunk-size", type=int, default=FOURIER_CHUNK_SIZE)

    parser.add_argument("--n-best-peaks", type=int, default=N_BEST_PEAKS)
    parser.add_argument("--peak-separation-widths", type=float, default=MIN_PEAK_SEPARATION_IN_WIDTHS)

    parser.add_argument("--ls-period", type=float, default=None)
    parser.add_argument("--ls-report", default=None)
    parser.add_argument("--ls-local-window-fraction", type=float, default=LS_LOCAL_WINDOW_FRACTION)
    parser.add_argument(
        "--global-search",
        action="store_true",
        help="Ignore the LS period and search the full Fourier period range.",
    )

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


def local_period_bounds(center_period, window_fraction, min_period, max_period):
    half_width = center_period * window_fraction
    local_min = max(min_period, center_period - half_width)
    local_max = min(max_period, center_period + half_width)

    if local_min >= local_max:
        raise ValueError("Local period window collapsed.")

    return local_min, local_max


def make_frequency_grid(min_period, max_period, n_frequencies):
    if n_frequencies < 5:
        raise ValueError("Frequency grid must contain at least 5 frequencies.")

    min_frequency = 1.0 / max_period
    max_frequency = 1.0 / min_period

    if min_frequency >= max_frequency:
        raise ValueError("Frequency range collapsed.")

    return np.linspace(min_frequency, max_frequency, n_frequencies)


def refine_peak_parabolic(freq_grid, power_grid, peak_idx):
    if peak_idx <= 0 or peak_idx >= len(freq_grid) - 1:
        return freq_grid[peak_idx]

    x0 = freq_grid[peak_idx]
    x = freq_grid[peak_idx - 1:peak_idx + 2] - x0
    y_power = power_grid[peak_idx - 1:peak_idx + 2]

    if not np.all(np.isfinite(y_power)):
        return freq_grid[peak_idx]

    a, b, _ = np.polyfit(x, y_power, 2)

    if a >= 0:
        return freq_grid[peak_idx]

    refined_frequency = x0 - b / (2.0 * a)

    if freq_grid[peak_idx - 1] <= refined_frequency <= freq_grid[peak_idx + 1]:
        return refined_frequency

    return freq_grid[peak_idx]


def top_fourier_peaks(frequency, power, n_peaks, min_separation):
    local_peak_mask = np.zeros_like(power, dtype=bool)
    local_peak_mask[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])

    candidate_indices = np.where(local_peak_mask)[0]

    if len(candidate_indices) == 0:
        candidate_indices = np.array([np.nanargmax(power)])

    candidate_indices = candidate_indices[np.argsort(power[candidate_indices])[::-1]]

    peak_indices = []
    for idx in candidate_indices:
        if all(abs(frequency[idx] - frequency[old_idx]) >= min_separation for old_idx in peak_indices):
            peak_indices.append(idx)

        if len(peak_indices) == n_peaks:
            break

    return peak_indices

# ============================================================
# WEIGHTED FOURIER CORE
# ============================================================
def make_weights(dy):
    if dy is None:
        return None

    weights = 1.0 / (dy * dy)
    median_weight = np.median(weights[np.isfinite(weights) & (weights > 0)])

    if np.isfinite(median_weight) and median_weight > 0:
        weights = weights / median_weight

    return weights


def solve_many(lhs, rhs):
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.array([
            np.linalg.lstsq(lhs_row, rhs_row, rcond=None)[0]
            for lhs_row, rhs_row in zip(lhs, rhs)
        ])


def fourier_power_grid(t_relative, y, dy, frequency_grid, chunk_size):
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive.")

    weights = make_weights(dy)
    if weights is None:
        weights = np.ones_like(y)

    s0 = np.sum(weights)
    wy = weights * y
    sy = np.sum(wy)
    ywy = np.sum(wy * y)
    chi2_constant = ywy - (sy * sy) / s0

    if not np.isfinite(chi2_constant) or chi2_constant <= 0:
        raise ValueError("Fourier analysis failed: signal variance is zero or invalid.")

    power = np.full(len(frequency_grid), np.nan)
    amplitude = np.full(len(frequency_grid), np.nan)

    for start in range(0, len(frequency_grid), chunk_size):
        stop = min(start + chunk_size, len(frequency_grid))
        freq_chunk = frequency_grid[start:stop]
        angle = 2.0 * np.pi * freq_chunk[:, np.newaxis] * t_relative[np.newaxis, :]

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        sc = cos_angle @ weights
        ss = sin_angle @ weights
        scc = (cos_angle * cos_angle) @ weights
        scs = (cos_angle * sin_angle) @ weights
        sss = (sin_angle * sin_angle) @ weights

        syc = cos_angle @ wy
        sys = sin_angle @ wy

        lhs = np.zeros((len(freq_chunk), 3, 3), dtype=float)
        lhs[:, 0, 0] = s0
        lhs[:, 0, 1] = sc
        lhs[:, 0, 2] = ss
        lhs[:, 1, 0] = sc
        lhs[:, 1, 1] = scc
        lhs[:, 1, 2] = scs
        lhs[:, 2, 0] = ss
        lhs[:, 2, 1] = scs
        lhs[:, 2, 2] = sss

        rhs = np.column_stack([
            np.full(len(freq_chunk), sy),
            syc,
            sys,
        ])

        coeff = solve_many(lhs, rhs)
        model_weighted_sum = np.sum(coeff * rhs, axis=1)
        chi2_model = ywy - model_weighted_sum
        explained = chi2_constant - chi2_model

        with np.errstate(divide="ignore", invalid="ignore"):
            chunk_power = explained / chi2_constant

        chunk_power = np.clip(chunk_power, 0.0, None)
        chunk_amplitude = np.sqrt(coeff[:, 1] * coeff[:, 1] + coeff[:, 2] * coeff[:, 2])

        power[start:stop] = chunk_power
        amplitude[start:stop] = chunk_amplitude

    return power, amplitude


def prepare_fourier_design(t_relative, dy, frequency_grid, chunk_size):
    weights = make_weights(dy)
    if weights is None:
        weights = np.ones_like(t_relative)

    s0 = np.sum(weights)
    chunks = []

    for start in range(0, len(frequency_grid), chunk_size):
        stop = min(start + chunk_size, len(frequency_grid))
        freq_chunk = frequency_grid[start:stop]
        angle = 2.0 * np.pi * freq_chunk[:, np.newaxis] * t_relative[np.newaxis, :]

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        sc = cos_angle @ weights
        ss = sin_angle @ weights
        scc = (cos_angle * cos_angle) @ weights
        scs = (cos_angle * sin_angle) @ weights
        sss = (sin_angle * sin_angle) @ weights

        lhs = np.zeros((len(freq_chunk), 3, 3), dtype=float)
        lhs[:, 0, 0] = s0
        lhs[:, 0, 1] = sc
        lhs[:, 0, 2] = ss
        lhs[:, 1, 0] = sc
        lhs[:, 1, 1] = scc
        lhs[:, 1, 2] = scs
        lhs[:, 2, 0] = ss
        lhs[:, 2, 1] = scs
        lhs[:, 2, 2] = sss

        try:
            lhs_inverse = np.linalg.inv(lhs)
        except np.linalg.LinAlgError:
            lhs_inverse = np.array([np.linalg.pinv(row) for row in lhs])

        chunks.append({
            "start": start,
            "stop": stop,
            "cos": cos_angle,
            "sin": sin_angle,
            "lhs_inverse": lhs_inverse,
        })

    return {
        "weights": weights,
        "s0": s0,
        "frequency_grid": frequency_grid,
        "chunks": chunks,
    }


def evaluate_fourier_design(design, y):
    weights = design["weights"]
    s0 = design["s0"]
    frequency_grid = design["frequency_grid"]

    wy = weights * y
    sy = np.sum(wy)
    ywy = np.sum(wy * y)
    chi2_constant = ywy - (sy * sy) / s0

    if not np.isfinite(chi2_constant) or chi2_constant <= 0:
        return np.full(len(frequency_grid), np.nan), np.full(len(frequency_grid), np.nan)

    power = np.full(len(frequency_grid), np.nan)
    amplitude = np.full(len(frequency_grid), np.nan)

    for chunk in design["chunks"]:
        start = chunk["start"]
        stop = chunk["stop"]
        cos_angle = chunk["cos"]
        sin_angle = chunk["sin"]
        lhs_inverse = chunk["lhs_inverse"]

        syc = cos_angle @ wy
        sys = sin_angle @ wy

        rhs = np.column_stack([
            np.full(stop - start, sy),
            syc,
            sys,
        ])

        coeff = np.einsum("fij,fj->fi", lhs_inverse, rhs)
        model_weighted_sum = np.sum(coeff * rhs, axis=1)
        chi2_model = ywy - model_weighted_sum
        explained = chi2_constant - chi2_model

        with np.errstate(divide="ignore", invalid="ignore"):
            chunk_power = explained / chi2_constant

        power[start:stop] = np.clip(chunk_power, 0.0, None)
        amplitude[start:stop] = np.sqrt(coeff[:, 1] * coeff[:, 1] + coeff[:, 2] * coeff[:, 2])

    return power, amplitude


def fourier_model_at_period(t_relative, y, dy, period):
    weights = make_weights(dy)
    if weights is None:
        weights = np.ones_like(y)

    angle = 2.0 * np.pi * t_relative / period
    design = np.column_stack([
        np.ones(len(t_relative)),
        np.cos(angle),
        np.sin(angle),
    ])

    lhs = design.T @ (weights[:, np.newaxis] * design)
    rhs = design.T @ (weights * y)

    try:
        coeff = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return design @ coeff

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
    raise ValueError("Too few valid data points for Fourier analysis.")

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

if args.chunk_size < 1:
    raise ValueError("chunk-size must be positive.")

if args.mc_grid_size < 5:
    raise ValueError("mc-grid-size must be at least 5.")

if args.max_period > time_span:
    print(
        "WARNING: max-period is longer than the data time span. "
        "Long-period Fourier peaks may be poorly constrained."
    )

t_relative = t - t.min()
y_centered = y - np.mean(y)
freq_resolution = 1.0 / time_span

# ============================================================
# GET LS PERIOD OR FALL BACK TO GLOBAL FOURIER SEARCH
# ============================================================
ls_period, ls_source = read_ls_period(args)

if ls_period is not None and not args.global_search:
    if args.min_period <= ls_period <= args.max_period:
        search_mode = "local around LS period"
        period_min, period_max = local_period_bounds(
            ls_period,
            args.ls_local_window_fraction,
            args.min_period,
            args.max_period
        )
        n_frequencies = args.n_frequencies_local
    else:
        search_mode = "global Fourier search; LS period outside requested range"
        period_min = args.min_period
        period_max = args.max_period
        n_frequencies = args.n_frequencies_global
else:
    search_mode = "global Fourier search"
    period_min = args.min_period
    period_max = args.max_period
    n_frequencies = args.n_frequencies_global

frequency = make_frequency_grid(period_min, period_max, n_frequencies)
period = 1.0 / frequency

# ============================================================
# OUTPUT FILE NAMES
# ============================================================
os.makedirs(args.output_dir, exist_ok=True)

base_name = os.path.splitext(os.path.basename(args.input))[0]
fourier_tag = f"{base_name}_fourier_MC{args.n_monte_carlo}_{args.min_period:.3f}-{args.max_period:.1f}d"

OUTPUT_SPECTRUM = os.path.join(args.output_dir, f"{fourier_tag}_spectrum.png")
OUTPUT_PHASE = os.path.join(args.output_dir, f"{fourier_tag}_phase_curve.png")
OUTPUT_PERIOD = os.path.join(args.output_dir, f"{fourier_tag}_best_period.txt")
OUTPUT_TOP_PEAKS = os.path.join(args.output_dir, f"{fourier_tag}_top_peaks.csv")
OUTPUT_SPECTRUM_CSV = os.path.join(args.output_dir, f"{fourier_tag}_spectrum.csv")
OUTPUT_MC_PERIODS = os.path.join(args.output_dir, f"{fourier_tag}_mc_periods.csv")
OUTPUT_MC_HIST = os.path.join(args.output_dir, f"{fourier_tag}_mc_period_hist.png")

# ============================================================
# MAIN FOURIER ANALYSIS
# ============================================================
print("=" * 70)
print("Running weighted Fourier analysis")
print(f"Input file                : {args.input}")
print(f"Output directory          : {args.output_dir}")
print(f"Search mode               : {search_mode}")
print(f"LS period source          : {ls_source if ls_source is not None else 'not available'}")
print(f"LS period                 : {ls_period if ls_period is not None else 'not available'}")
print(f"Fourier period range      : {period_min:.10f} .. {period_max:.10f} days")
print(f"Fourier frequencies       : {len(frequency)}")
print(f"Frequency resolution ~    : {freq_resolution:.10f} 1/day")
print(f"Points used               : {len(t)}")
print(f"Time span                 : {time_span:.10f} days")
print("=" * 70)

power, amplitude = fourier_power_grid(
    t_relative,
    y,
    dy,
    frequency,
    args.chunk_size
)

if not np.any(np.isfinite(power)):
    raise ValueError("Fourier analysis failed: all power values are NaN.")

# ============================================================
# BEST PERIOD
# ============================================================
best_idx = np.nanargmax(power)
best_frequency = refine_peak_parabolic(frequency, power, best_idx)
best_period = 1.0 / best_frequency
best_power = power[best_idx]
best_amplitude = amplitude[best_idx]

# ============================================================
# TOP PEAKS
# ============================================================
min_peak_separation = args.peak_separation_widths * freq_resolution
peak_indices = top_fourier_peaks(
    frequency,
    power,
    args.n_best_peaks,
    min_peak_separation
)

top_peaks_data = []
for i, idx in enumerate(peak_indices, start=1):
    top_peaks_data.append({
        "rank": i,
        "period_days": 1.0 / frequency[idx],
        "frequency_per_day": frequency[idx],
        "fourier_power": power[idx],
        "semi_amplitude": amplitude[idx],
    })

pd.DataFrame(top_peaks_data).to_csv(OUTPUT_TOP_PEAKS, index=False)

pd.DataFrame({
    "period_days": period,
    "frequency_per_day": frequency,
    "fourier_power": power,
    "semi_amplitude": amplitude,
}).sort_values("period_days").to_csv(OUTPUT_SPECTRUM_CSV, index=False)

# ============================================================
# MONTE CARLO AROUND THE BEST FOURIER PERIOD
# ============================================================
rng = np.random.default_rng(args.seed)
mc_periods = []

mc_min_period, mc_max_period = local_period_bounds(
    best_period,
    args.mc_window_fraction,
    args.min_period,
    args.max_period
)

mc_frequency_grid = make_frequency_grid(mc_min_period, mc_max_period, args.mc_grid_size)
mc_design = prepare_fourier_design(t_relative, dy, mc_frequency_grid, args.chunk_size)

if dy is not None:
    mc_noise_sigma = dy
    mc_noise_level = np.nanmedian(mc_noise_sigma)
    mc_noise_source = f"measured photometric errors from '{err_col}'"
else:
    best_model = fourier_model_at_period(t_relative, y, None, best_period)
    residuals = y - best_model
    mc_noise_level = robust_scatter(residuals)
    mc_noise_sigma = np.full_like(y, mc_noise_level)
    mc_noise_source = "estimated from residual scatter around the best Fourier sine model"

print("=" * 70)
print("Running Fourier Monte Carlo")
print(f"Monte Carlo iterations    : {args.n_monte_carlo}")
print(f"MC noise model            : {mc_noise_source}")
print(f"MC typical noise level    : {mc_noise_level:.10f}")
print(f"MC period range           : {mc_min_period:.10f} .. {mc_max_period:.10f} days")
print(f"MC frequency grid size    : {len(mc_frequency_grid)}")
print("=" * 70)

for i in range(args.n_monte_carlo):
    y_mc = y + rng.normal(loc=0.0, scale=mc_noise_sigma, size=len(y))

    power_mc, _ = evaluate_fourier_design(mc_design, y_mc)

    if not np.any(np.isfinite(power_mc)):
        continue

    best_idx_mc = np.nanargmax(power_mc)
    best_freq_mc = refine_peak_parabolic(mc_frequency_grid, power_mc, best_idx_mc)
    best_period_mc = 1.0 / best_freq_mc
    mc_periods.append(best_period_mc)

    if (i + 1) % 100 == 0 or (i + 1) == args.n_monte_carlo:
        print(f"Monte Carlo: {i + 1}/{args.n_monte_carlo}")

mc_periods = np.array(mc_periods)

if len(mc_periods) == 0:
    raise ValueError("Fourier Monte Carlo failed: no valid MC periods were found.")

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
print("FINAL FOURIER RESULTS")
print(f"Best Fourier period            : {best_period:.10f} days")
print(f"Best Fourier frequency         : {best_frequency:.10f} 1/day")
print(f"Best Fourier power             : {best_power:.10f}")
print(f"Best semi-amplitude            : {best_amplitude:.10f}")
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

print("\nTop peaks from the Fourier search:")
for row in top_peaks_data:
    print(
        f"{row['rank']:2d}. "
        f"Period = {row['period_days']:.10f} days, "
        f"Frequency = {row['frequency_per_day']:.10f} 1/day, "
        f"Power = {row['fourier_power']:.10f}, "
        f"Semi-amplitude = {row['semi_amplitude']:.10f}"
    )

# ============================================================
# SAVE TEXT REPORT
# ============================================================
with open(OUTPUT_PERIOD, "w", encoding="utf-8") as f:
    f.write("=== FOURIER RESULTS ===\n")
    f.write("Method: weighted sinusoidal Fourier fit at each trial frequency\n")
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
    f.write(f"Frequency resolution approx (1/day): {freq_resolution:.10f}\n")
    f.write(f"Min period searched (days): {period_min:.10f}\n")
    f.write(f"Max period searched (days): {period_max:.10f}\n")
    f.write(f"Fourier frequencies: {len(frequency)}\n\n")

    f.write(f"Best Fourier period (days): {best_period:.10f}\n")
    f.write(f"Best Fourier frequency (1/day): {best_frequency:.10f}\n")
    f.write(f"Best Fourier power: {best_power:.10f}\n")
    f.write(f"Best semi-amplitude: {best_amplitude:.10f}\n")
    if ls_period is not None:
        f.write(f"Difference from LS period (days): {best_period - ls_period:.10f}\n")
    f.write("\n")

    f.write("=== MONTE CARLO SETTINGS ===\n")
    f.write(f"Monte Carlo iterations: {args.n_monte_carlo}\n")
    f.write(f"Random seed: {args.seed}\n")
    f.write(f"MC period range (days): {mc_min_period:.10f} .. {mc_max_period:.10f}\n")
    f.write(f"MC frequency grid size: {len(mc_frequency_grid)}\n")
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
# PLOT: FOURIER SPECTRUM
# ============================================================
sort_idx = np.argsort(period)
period_sorted = period[sort_idx]
power_sorted = power[sort_idx]

plt.figure(figsize=(11, 6))
plt.plot(period_sorted, power_sorted, linewidth=1.2)
plt.axvline(best_period, linestyle="--", alpha=0.9, label=f"Best Fourier = {best_period:.6f} d")
if ls_period is not None and period_min <= ls_period <= period_max:
    plt.axvline(ls_period, linestyle=":", alpha=0.9, label=f"LS period = {ls_period:.6f} d")
plt.xlabel("Period (days)")
plt.ylabel("Fourier power")
plt.title("Weighted Fourier spectrum")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_SPECTRUM, dpi=300)

if args.show_plots:
    plt.show()

plt.close()

# ============================================================
# PLOT: PHASE-FOLDED CURVE
# ============================================================
phase = (t_relative % best_period) / best_period

plt.figure(figsize=(9, 6))
plt.scatter(phase, y, s=18, alpha=0.8, label="Data")
plt.scatter(phase + 1.0, y, s=18, alpha=0.8, label="Repeated phase")
plt.xlabel("Phase")
plt.ylabel(signal_label)
plt.title(f"Phase-folded curve (Fourier period = {best_period:.6f} d)")
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
plt.axvline(best_period, linestyle="--", label=f"Best Fourier = {best_period:.10f} d")
plt.axvline(period_mean_mc, linestyle="-.", label=f"MC mean = {period_mean_mc:.10f} d")
plt.axvline(period_p16, linestyle=":", label=f"P16 = {period_p16:.10f} d")
plt.axvline(period_p84, linestyle=":", label=f"P84 = {period_p84:.10f} d")
plt.xlabel("Period (days)")
plt.ylabel("Count")
plt.title("Monte Carlo distribution of Fourier periods")

axis_values = np.concatenate([
    mc_periods[np.isfinite(mc_periods)],
    np.array([best_period, period_mean_mc, period_p16, period_p84], dtype=float),
])
axis_values = axis_values[np.isfinite(axis_values)]

if len(axis_values) > 0:
    axis_min = np.min(axis_values)
    axis_max = np.max(axis_values)
    axis_width = axis_max - axis_min

    if not np.isfinite(axis_width) or axis_width <= 0:
        axis_width = max(best_period * 1e-8, 1e-10)

    plt.xlim(axis_min - 0.15 * axis_width, axis_max + 0.15 * axis_width)

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
    OUTPUT_SPECTRUM,
    OUTPUT_PHASE,
    OUTPUT_MC_HIST,
    OUTPUT_PERIOD,
    OUTPUT_TOP_PEAKS,
    OUTPUT_SPECTRUM_CSV,
    OUTPUT_MC_PERIODS,
]

removed_old_outputs = clean_old_outputs(
    args.output_dir,
    current_outputs,
    [
        "*_fourier_MC*_spectrum.png",
        "*_fourier_MC*_phase_curve.png",
        "*_fourier_MC*_mc_period_hist.png",
        "*_fourier_MC*_best_period.txt",
        "*_fourier_MC*_top_peaks.csv",
        "*_fourier_MC*_spectrum.csv",
        "*_fourier_MC*_mc_periods.csv",
    ],
)

if removed_old_outputs > 0:
    print(f"\nRemoved old Fourier output files: {removed_old_outputs}")

print("\nSaved files:")
for file_path in current_outputs:
    print(f"- {os.path.relpath(file_path, PROJECT_DIR)}")
