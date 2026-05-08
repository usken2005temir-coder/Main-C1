import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

try:
    from astropy.timeseries import LombScargle
except ImportError:
    LombScargle = None


# ============================================================
# DEFAULT SETTINGS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(PROJECT_DIR, "clean_data.csv")
OUTPUT_DIR = SCRIPT_DIR

MIN_PERIOD = 0.05
MAX_PERIOD = 100.0

N_GRID = 5000
N_PDM_GRID = 1501
N_PHASE_BINS = 20
LS_SAMPLES_PER_PEAK = 20
SHOW_PLOTS = True

# Each method first performs an independent global search. Then PDM and Fourier
# refine around their own candidates, not around one fixed example star period.
LOCAL_WINDOW_FRACTION = 0.01


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
        description=(
            "Smart variable-star period analysis. The script compares LS, PDM and "
            "weighted Fourier diagnostics, recommends the main method, and writes "
            "plots with automatic period units."
        )
    )

    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)

    parser.add_argument("--time-col", default=None)
    parser.add_argument("--signal-col", default=None)
    parser.add_argument("--error-col", default=None)

    parser.add_argument("--min-period", type=float, default=MIN_PERIOD)
    parser.add_argument("--max-period", type=float, default=MAX_PERIOD)
    parser.add_argument("--n-grid", type=int, default=N_GRID)
    parser.add_argument("--n-pdm-grid", type=int, default=N_PDM_GRID)
    parser.add_argument("--n-phase-bins", type=int, default=N_PHASE_BINS)
    parser.add_argument("--ls-samples-per-peak", type=int, default=LS_SAMPLES_PER_PEAK)
    parser.add_argument("--local-window-fraction", type=float, default=LOCAL_WINDOW_FRACTION)

    parser.add_argument("--show-plots", action="store_true", default=SHOW_PLOTS)
    parser.add_argument("--no-show-plots", action="store_false", dest="show_plots")

    return parser.parse_args()


# ============================================================
# COLUMN DETECTION
# ============================================================
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
        ["mag", "magnitude", "vmag", "gmag", "rmag", "imag", "v/r", "vr", "flux", "brightness", "v", "r", "i"],
        forbidden_aliases=["error", "err", "sigma", "uncertainty", "limit", "lambda", "wave", "wavelength"],
    )

    if time_col is None:
        raise ValueError(f"No time column found. Available columns: {columns}")
    if signal_col is None:
        raise ValueError(f"No magnitude/flux column found. Available columns: {columns}")

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


# ============================================================
# DATA LOADING
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


def load_light_curve(args):
    df = read_input_table(args.input)
    if df.empty:
        raise ValueError("Input file is empty.")

    time_col, signal_col, err_col = choose_columns(df, args)
    signal_label = display_label_from_dataframe(df, signal_col)

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[signal_col], errors="coerce").to_numpy()

    if err_col is not None:
        dy = pd.to_numeric(df[err_col], errors="coerce").to_numpy()
    else:
        dy = np.full(len(t), np.nan)

    if np.any(np.isfinite(dy) & (dy > 0)):
        finite_dy = dy[np.isfinite(dy) & (dy > 0)]
        fallback_dy = np.nanmedian(finite_dy)
        dy = np.where(np.isfinite(dy) & (dy > 0), dy, fallback_dy)
    else:
        dy = np.ones(len(t), dtype=float)
        err_col = None

    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    t = t[mask]
    y = y[mask]
    dy = dy[mask]

    if len(t) < 10:
        raise ValueError("Too few valid data points for smart period analysis.")

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    dy = dy[order]

    return df, t, y, dy, time_col, signal_col, signal_label, err_col


# ============================================================
# PERIOD UNITS
# ============================================================
def choose_period_unit(period_days):
    period_days = abs(float(period_days))

    if period_days < 0.1:
        return {
            "name": "minutes",
            "label": "Period (minutes)",
            "factor": 24.0 * 60.0,
            "format": "{:.4f} min",
        }

    if period_days < 365.25:
        return {
            "name": "days",
            "label": "Period (days)",
            "factor": 1.0,
            "format": "{:.8f} d",
        }

    return {
        "name": "years",
        "label": "Period (years)",
        "factor": 1.0 / 365.25,
        "format": "{:.6f} yr",
    }


def format_period(period_days, unit_info=None):
    if unit_info is None:
        unit_info = choose_period_unit(period_days)
    return unit_info["format"].format(period_days * unit_info["factor"])


# ============================================================
# MODEL HELPERS
# ============================================================
def weighted_mean(y, dy):
    w = 1.0 / dy**2
    return np.sum(w * y) / np.sum(w)


def robust_scatter(values):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scatter = 1.4826 * mad

    if not np.isfinite(scatter) or scatter <= 0:
        scatter = np.std(values, ddof=1) if len(values) > 1 else 1.0

    if not np.isfinite(scatter) or scatter <= 0:
        scatter = 1.0

    return scatter


def fit_weighted_harmonic(t, y, dy, period, n_harmonics=1):
    phase_time = t - np.min(t)
    omega = 2.0 * np.pi / period

    columns = [np.ones_like(t)]
    for harmonic in range(1, n_harmonics + 1):
        columns.append(np.cos(harmonic * omega * phase_time))
        columns.append(np.sin(harmonic * omega * phase_time))

    x = np.column_stack(columns)
    sqrt_w = 1.0 / dy

    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w

    coeffs, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
    model = x @ coeffs
    chi2 = np.sum(((y - model) / dy) ** 2)

    return coeffs, model, chi2


def refine_peak_parabolic(x_grid, y_grid, peak_idx, mode="max"):
    if peak_idx <= 0 or peak_idx >= len(x_grid) - 1:
        return x_grid[peak_idx]

    x0 = x_grid[peak_idx]
    x = x_grid[peak_idx - 1:peak_idx + 2] - x0
    y = y_grid[peak_idx - 1:peak_idx + 2]
    a, b, _ = np.polyfit(x, y, 2)

    if mode == "max" and a >= 0:
        return x_grid[peak_idx]
    if mode == "min" and a <= 0:
        return x_grid[peak_idx]

    refined = x0 - b / (2.0 * a)
    if x_grid[peak_idx - 1] <= refined <= x_grid[peak_idx + 1]:
        return refined

    return x_grid[peak_idx]


# ============================================================
# METHOD: LOMB-SCARGLE
# ============================================================
def run_lomb_scargle(t, y, dy, min_period, max_period, samples_per_peak):
    if LombScargle is None:
        return None

    min_frequency = 1.0 / max_period
    max_frequency = 1.0 / min_period

    y_centered = y - weighted_mean(y, dy)
    ls = LombScargle(t, y_centered, dy=dy)
    frequency, power = ls.autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=samples_per_peak,
    )

    best_idx = int(np.argmax(power))
    best_frequency = refine_peak_parabolic(frequency, power, best_idx, mode="max")
    best_period = 1.0 / best_frequency

    return {
        "name": "Lomb-Scargle",
        "period": best_period,
        "frequency": best_frequency,
        "score": float(np.nanmax(power)),
        "x": 1.0 / frequency,
        "y": power,
        "score_label": "power",
    }


# ============================================================
# METHOD: WEIGHTED FOURIER
# ============================================================
def run_fourier_local(t, y, dy, reference_period, min_period, max_period, n_grid, local_window_fraction):
    if reference_period is None or not np.isfinite(reference_period):
        period_min = min_period
        period_max = max_period
    else:
        half_width = max(reference_period * local_window_fraction, min_period * 0.02)
        period_min = max(min_period, reference_period - half_width)
        period_max = min(max_period, reference_period + half_width)

    frequency = np.linspace(1.0 / period_max, 1.0 / period_min, n_grid)
    periods = 1.0 / frequency

    y0 = weighted_mean(y, dy)
    chi2_const = np.sum(((y - y0) / dy) ** 2)
    powers = np.empty_like(frequency)

    for i, freq in enumerate(frequency):
        period = 1.0 / freq
        _, _, chi2_model = fit_weighted_harmonic(t, y, dy, period, n_harmonics=1)
        powers[i] = max(0.0, (chi2_const - chi2_model) / chi2_const)

    best_idx = int(np.argmax(powers))
    best_frequency = refine_peak_parabolic(frequency, powers, best_idx, mode="max")
    best_period = 1.0 / best_frequency

    return {
        "name": "Weighted Fourier",
        "period": best_period,
        "frequency": best_frequency,
        "score": float(np.nanmax(powers)),
        "x": periods,
        "y": powers,
        "score_label": "power",
    }


# ============================================================
# METHOD: PDM
# ============================================================
def pdm_theta(t, y, period, n_bins):
    phase = (t / period) % 1.0
    bin_index = np.floor(phase * n_bins).astype(int)
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    total_var = np.var(y, ddof=1)
    if total_var <= 0 or not np.isfinite(total_var):
        return np.nan

    numerator = 0.0
    points_used = 0

    for bin_id in range(n_bins):
        values = y[bin_index == bin_id]
        if len(values) >= 2:
            numerator += (len(values) - 1) * np.var(values, ddof=1)
            points_used += len(values) - 1

    if points_used <= 0:
        return np.nan

    return numerator / ((len(y) - 1) * total_var)


def run_pdm_local(t, y, reference_period, min_period, max_period, n_grid, n_bins, local_window_fraction):
    if reference_period is None or not np.isfinite(reference_period):
        frequency = np.linspace(1.0 / max_period, 1.0 / min_period, n_grid)
        periods = 1.0 / frequency
    else:
        half_width = max(reference_period * local_window_fraction, min_period * 0.02)
        period_min = max(min_period, reference_period - half_width)
        period_max = min(max_period, reference_period + half_width)
        periods = np.linspace(period_min, period_max, n_grid)

    theta = np.array([pdm_theta(t, y, period, n_bins) for period in periods])

    if not np.any(np.isfinite(theta)):
        return None

    best_idx = int(np.nanargmin(theta))
    best_period = refine_peak_parabolic(periods, theta, best_idx, mode="min")

    return {
        "name": "PDM",
        "period": best_period,
        "frequency": 1.0 / best_period,
        "score": float(np.nanmin(theta)),
        "x": periods,
        "y": theta,
        "score_label": "theta",
    }


# ============================================================
# DATA DIAGNOSTICS
# ============================================================
def phase_coverage(t, period, n_bins):
    phase = (t / period) % 1.0
    counts, _ = np.histogram(phase, bins=n_bins, range=(0.0, 1.0))
    return np.count_nonzero(counts > 0) / n_bins


def time_gap_diagnostics(t):
    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        return {
            "median_dt": np.nan,
            "max_gap": np.nan,
            "gap_ratio": np.inf,
        }

    median_dt = float(np.median(dt))
    max_gap = float(np.max(dt))
    gap_ratio = max_gap / median_dt if median_dt > 0 else np.inf

    return {
        "median_dt": median_dt,
        "max_gap": max_gap,
        "gap_ratio": gap_ratio,
    }


def sinusoidality_diagnostics(t, y, dy, period):
    y0 = weighted_mean(y, dy)
    chi2_const = np.sum(((y - y0) / dy) ** 2)

    _, model_1, chi2_1 = fit_weighted_harmonic(t, y, dy, period, n_harmonics=1)
    _, model_2, chi2_2 = fit_weighted_harmonic(t, y, dy, period, n_harmonics=2)

    sine_power = max(0.0, (chi2_const - chi2_1) / chi2_const)
    harmonic_gain = max(0.0, (chi2_1 - chi2_2) / chi2_const)
    harmonic_gain_fraction = harmonic_gain / max(sine_power, 1e-12)

    return {
        "sine_power": float(sine_power),
        "harmonic_gain": float(harmonic_gain),
        "harmonic_gain_fraction": float(harmonic_gain_fraction),
        "sine_model": model_1,
        "two_harmonic_model": model_2,
    }


def recommend_method(gap_ratio, coverage, sine_power, harmonic_gain_fraction):
    gappy = gap_ratio > 20.0 or coverage < 0.75
    non_sinusoidal = harmonic_gain_fraction > 0.25 or sine_power < 0.15

    if non_sinusoidal:
        return {
            "method": "PDM",
            "reason": (
                "The folded curve is not well described by a simple sine model, "
                "so phase-dispersion minimization is the safest main criterion."
            ),
        }

    if gappy:
        return {
            "method": "Lomb-Scargle",
            "reason": (
                "The time sampling has large gaps or incomplete phase coverage, "
                "so Lomb-Scargle is the best main method for uneven observations."
            ),
        }

    return {
        "method": "Weighted Fourier",
        "reason": (
            "The sampling is reasonably continuous and the signal is close to sinusoidal, "
            "so a weighted Fourier fit is the clean main model."
        ),
    }


def choose_adopted_period(results, recommendation):
    finite_periods = [
        result["period"]
        for result in results.values()
        if result is not None and np.isfinite(result["period"])
    ]

    recommended_result = results.get(recommendation["method"])
    relative_spread = np.nan

    if len(finite_periods) >= 2:
        median_period = float(np.median(finite_periods))
        relative_spread = (max(finite_periods) - min(finite_periods)) / median_period

    if recommended_result is not None:
        return float(recommended_result["period"]), recommendation["method"], relative_spread

    if finite_periods:
        return float(finite_periods[0]), "fallback", relative_spread

    raise ValueError("No valid period estimate was produced.")


# ============================================================
# PLOTS
# ============================================================
def plot_method_comparison(results, adopted_period, adopted_source, output_path, show_plots):
    unit_info = choose_period_unit(adopted_period)
    names = []
    period_values = []
    period_days = []

    for key in ["Lomb-Scargle", "PDM", "Weighted Fourier"]:
        result = results.get(key)
        if result is not None and np.isfinite(result["period"]):
            names.append(key.replace("Weighted ", ""))
            period_values.append(result["period"] * unit_info["factor"])
            period_days.append(result["period"])

    adopted_value = adopted_period * unit_info["factor"]

    plt.figure(figsize=(10, 5.8))
    plt.scatter(names, period_values, s=95, zorder=3)
    plt.axhline(
        adopted_value,
        linestyle="--",
        alpha=0.85,
        label=f"Adopted period ({adopted_source})"
    )

    for i, (value, period_day) in enumerate(zip(period_values, period_days)):
        delta_seconds = (period_day - adopted_period) * 24.0 * 60.0 * 60.0
        plt.annotate(
            f"{value:.4f}\nDelta {delta_seconds:+.3f} s",
            (i, value),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if period_values:
        spread = max(period_values) - min(period_values)
        y_margin = max(spread * 0.75, abs(adopted_value) * 1e-6, 1e-6)
        plt.ylim(min(period_values + [adopted_value]) - y_margin, max(period_values + [adopted_value]) + y_margin * 1.8)

    plt.ylabel(unit_info["label"])
    plt.title("Smart method comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.savefig(output_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close()


def plot_phase_curve(t, y, dy, period, output_path, signal_label, show_plots):
    unit_info = choose_period_unit(period)
    phase = (t / period) % 1.0

    order = np.argsort(phase)
    phase_sorted = phase[order]

    _, model, _ = fit_weighted_harmonic(t, y, dy, period, n_harmonics=1)
    model_sorted = model[order]

    plt.figure(figsize=(9, 6))
    plt.scatter(phase, y, s=16, alpha=0.75, label="Data")
    plt.scatter(phase + 1.0, y, s=16, alpha=0.75, label="Repeated phase")
    plt.plot(phase_sorted, model_sorted, color="black", linewidth=2, label="Weighted sine fit")
    plt.plot(phase_sorted + 1.0, model_sorted, color="black", linewidth=2)
    plt.xlabel("Phase")
    plt.ylabel(signal_label)
    plt.title(f"Smart phase-folded curve, P = {format_period(period, unit_info)}")
    plt.xlim(0, 2)
    plt.grid(True, alpha=0.3)
    if is_magnitude_label(signal_label):
        plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close()


def plot_diagnostics(diagnostics, output_path, show_plots):
    labels = [
        "Phase coverage",
        "Sine power",
        "2nd harmonic gain",
        "Gap pressure",
    ]

    gap_pressure = min(1.0, diagnostics["gap_ratio"] / 50.0)
    values = [
        diagnostics["coverage"],
        diagnostics["sine_power"],
        min(1.0, diagnostics["harmonic_gain_fraction"]),
        gap_pressure,
    ]

    colors = ["#50d6c8", "#a8df6a", "#f3b15f", "#b9a6ff"]

    plt.figure(figsize=(9, 5.4))
    plt.bar(labels, values, color=colors, alpha=0.85)
    plt.ylim(0, 1.05)
    plt.ylabel("Diagnostic score")
    plt.title("Smart decision diagnostics")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close()


def plot_periodogram_panels(results, adopted_period, output_path, show_plots):
    unit_info = choose_period_unit(adopted_period)

    available = [
        results[key]
        for key in ["Lomb-Scargle", "PDM", "Weighted Fourier"]
        if results.get(key) is not None
    ]

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 3.3 * len(available)), sharex=False)
    if len(available) == 1:
        axes = [axes]

    for ax, result in zip(axes, available):
        x = result["x"] * unit_info["factor"]
        y_values = result["y"]
        sort_idx = np.argsort(x)

        ax.plot(x[sort_idx], y_values[sort_idx], linewidth=1.2)
        ax.axvline(adopted_period * unit_info["factor"], linestyle="--", alpha=0.8)
        ax.set_ylabel(result["score_label"])
        ax.set_title(result["name"])
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    axes[-1].set_xlabel(unit_info["label"])
    fig.suptitle("Smart periodograms in automatic units", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)


# ============================================================
# REPORT
# ============================================================
def write_report(output_path, args, columns, results, diagnostics, recommendation, adopted_period, adopted_source):
    unit_info = choose_period_unit(adopted_period)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== SMART VARIABLE-STAR PERIOD ANALYSIS ===\n")
        f.write(f"Input file: {args.input}\n")
        f.write(f"Time column: {columns['time']}\n")
        f.write(f"Signal column: {columns['signal']}\n")
        f.write(f"Signal label: {columns['signal_label']}\n")
        f.write(f"Error column: {columns['error'] if columns['error'] is not None else 'none'}\n")
        f.write(f"Search range: {args.min_period} .. {args.max_period} days\n\n")

        f.write("=== METHOD RESULTS ===\n")
        for key in ["Lomb-Scargle", "PDM", "Weighted Fourier"]:
            result = results.get(key)
            if result is None:
                f.write(f"{key}: not available\n")
                continue
            f.write(
                f"{key}: period = {result['period']:.10f} days "
                f"({format_period(result['period'], unit_info)}), "
                f"{result['score_label']} = {result['score']:.10f}\n"
            )

        f.write("\n=== DATA DIAGNOSTICS ===\n")
        f.write(f"Median time step: {diagnostics['median_dt']:.10f} days\n")
        f.write(f"Maximum time gap: {diagnostics['max_gap']:.10f} days\n")
        f.write(f"Gap ratio: {diagnostics['gap_ratio']:.4f}\n")
        f.write(f"Phase coverage: {diagnostics['coverage']:.4f}\n")
        f.write(f"Sine power: {diagnostics['sine_power']:.4f}\n")
        f.write(f"Second harmonic gain: {diagnostics['harmonic_gain']:.4f}\n")
        f.write(f"Second harmonic gain fraction: {diagnostics['harmonic_gain_fraction']:.4f}\n\n")
        f.write(f"Noise model: {diagnostics['noise_source']}\n")
        f.write(f"Typical noise level: {diagnostics['noise_level']:.10f}\n\n")

        f.write("=== SMART RECOMMENDATION ===\n")
        f.write(f"Recommended main method: {recommendation['method']}\n")
        f.write(f"Reason: {recommendation['reason']}\n")
        f.write(f"Adopted period source: {adopted_source}\n")
        f.write(f"Adopted period: {adopted_period:.10f} days\n")
        f.write(f"Adopted period in automatic units: {format_period(adopted_period, unit_info)}\n")
        f.write(f"Automatic period unit for plots: {unit_info['name']}\n\n")

        f.write("=== METHOD LOGIC ===\n")
        f.write("If the data have large gaps or incomplete phase coverage, LS is preferred.\n")
        f.write("If the sampling is continuous and the signal is sine-like, weighted Fourier is preferred.\n")
        f.write("If the phase curve is not sine-like, PDM is preferred.\n")
        f.write("Wavelet analysis remains a stability check in time, not the only final period source.\n")


# ============================================================
# MAIN
# ============================================================
args = parse_args()

if args.min_period <= 0 or args.max_period <= 0:
    raise ValueError("Minimum and maximum period must be positive.")
if args.min_period >= args.max_period:
    raise ValueError("Minimum period must be smaller than maximum period.")

os.makedirs(args.output_dir, exist_ok=True)

_, t, y, dy, time_col, signal_col, signal_label, err_col = load_light_curve(args)

print("=" * 72)
print("Running smart variable-star period analysis")
print(f"Input file       : {args.input}")
print(f"Output directory : {args.output_dir}")
print(f"Points used      : {len(t)}")
print(f"Time column      : {time_col}")
print(f"Signal column    : {signal_col}")
print(f"Signal label     : {signal_label}")
print(f"Error column     : {err_col if err_col is not None else 'not found'}")
print("=" * 72)

ls_result = run_lomb_scargle(t, y, dy, args.min_period, args.max_period, args.ls_samples_per_peak)
reference_period = ls_result["period"] if ls_result is not None else None

pdm_global_result = run_pdm_local(
    t,
    y,
    None,
    args.min_period,
    args.max_period,
    args.n_pdm_grid,
    args.n_phase_bins,
    args.local_window_fraction,
)
pdm_result = (
    run_pdm_local(
        t,
        y,
        pdm_global_result["period"],
        args.min_period,
        args.max_period,
        args.n_pdm_grid,
        args.n_phase_bins,
        args.local_window_fraction,
    )
    if pdm_global_result is not None
    else None
)

fourier_global_result = run_fourier_local(
    t,
    y,
    dy,
    None,
    args.min_period,
    args.max_period,
    args.n_grid,
    args.local_window_fraction,
)
fourier_result = (
    run_fourier_local(
        t,
        y,
        dy,
        fourier_global_result["period"],
        args.min_period,
        args.max_period,
        args.n_grid,
        args.local_window_fraction,
    )
    if fourier_global_result is not None
    else None
)

independent_results = {
    "Lomb-Scargle": ls_result,
    "PDM": pdm_result,
    "Weighted Fourier": fourier_result,
}

if reference_period is None:
    reference_period = fourier_result["period"] if fourier_result is not None else pdm_result["period"]

gap_info = time_gap_diagnostics(t)
coverage = phase_coverage(t, reference_period, args.n_phase_bins)
sine_info = sinusoidality_diagnostics(t, y, dy, reference_period)

if err_col is None:
    noise_source = "estimated from residual scatter around the reference-period sine model"
    noise_level = robust_scatter(y - sine_info["sine_model"])
else:
    noise_source = f"measured photometric errors from '{err_col}'"
    noise_level = np.nanmedian(dy)

diagnostics = {
    **gap_info,
    "coverage": coverage,
    "sine_power": sine_info["sine_power"],
    "harmonic_gain": sine_info["harmonic_gain"],
    "harmonic_gain_fraction": sine_info["harmonic_gain_fraction"],
    "noise_source": noise_source,
    "noise_level": noise_level,
}

recommendation = recommend_method(
    diagnostics["gap_ratio"],
    diagnostics["coverage"],
    diagnostics["sine_power"],
    diagnostics["harmonic_gain_fraction"],
)
adopted_period, adopted_source, independent_relative_spread = choose_adopted_period(independent_results, recommendation)
unit_info = choose_period_unit(adopted_period)

# After the main method chooses the adopted period, validate that period with the
# other diagnostics in a narrow local window. This keeps the smart output readable
# and avoids confusing aliases from broad global scans.
pdm_validation_result = run_pdm_local(
    t,
    y,
    adopted_period,
    args.min_period,
    args.max_period,
    args.n_pdm_grid,
    args.n_phase_bins,
    args.local_window_fraction,
)
fourier_validation_result = run_fourier_local(
    t,
    y,
    dy,
    adopted_period,
    args.min_period,
    args.max_period,
    args.n_grid,
    args.local_window_fraction,
)

results = {
    "Lomb-Scargle": ls_result,
    "PDM": pdm_validation_result,
    "Weighted Fourier": fourier_validation_result,
}

validation_periods = [
    result["period"]
    for result in results.values()
    if result is not None and np.isfinite(result["period"])
]
if len(validation_periods) >= 2:
    validation_median = float(np.median(validation_periods))
    relative_spread = (max(validation_periods) - min(validation_periods)) / validation_median
else:
    relative_spread = np.nan

OUTPUT_REPORT = os.path.join(args.output_dir, "smart_period_report.txt")
OUTPUT_COMPARISON = os.path.join(args.output_dir, "smart_method_comparison.png")
OUTPUT_PHASE = os.path.join(args.output_dir, "smart_phase_curve.png")
OUTPUT_DIAGNOSTICS = os.path.join(args.output_dir, "smart_diagnostics.png")
OUTPUT_PERIODOGRAMS = os.path.join(args.output_dir, "smart_periodograms.png")

write_report(
    OUTPUT_REPORT,
    args,
    {"time": time_col, "signal": signal_col, "signal_label": signal_label, "error": err_col},
    results,
    diagnostics,
    recommendation,
    adopted_period,
    adopted_source,
)
plot_method_comparison(results, adopted_period, adopted_source, OUTPUT_COMPARISON, args.show_plots)
plot_phase_curve(t, y, dy, adopted_period, OUTPUT_PHASE, signal_label, args.show_plots)
plot_diagnostics(diagnostics, OUTPUT_DIAGNOSTICS, args.show_plots)
plot_periodogram_panels(results, adopted_period, OUTPUT_PERIODOGRAMS, args.show_plots)

current_outputs = [
    OUTPUT_REPORT,
    OUTPUT_COMPARISON,
    OUTPUT_PHASE,
    OUTPUT_DIAGNOSTICS,
    OUTPUT_PERIODOGRAMS,
]

removed_old_outputs = clean_old_outputs(
    args.output_dir,
    current_outputs,
    [
        "smart_period_report.txt",
        "smart_method_comparison.png",
        "smart_phase_curve.png",
        "smart_diagnostics.png",
        "smart_periodograms.png",
    ],
)

if removed_old_outputs > 0:
    print(f"\nRemoved old smart output files: {removed_old_outputs}")

print("\nMethod results:")
for key in ["Lomb-Scargle", "PDM", "Weighted Fourier"]:
    result = results.get(key)
    if result is None:
        print(f"- {key}: not available")
    else:
        print(f"- {key}: {result['period']:.10f} days = {format_period(result['period'], unit_info)}")

print("\nSmart diagnostics:")
print(f"- Gap ratio                  : {diagnostics['gap_ratio']:.4f}")
print(f"- Phase coverage             : {diagnostics['coverage']:.4f}")
print(f"- Sine power                 : {diagnostics['sine_power']:.4f}")
print(f"- Second harmonic gain frac. : {diagnostics['harmonic_gain_fraction']:.4f}")
print(f"- Noise model                : {diagnostics['noise_source']}")
print(f"- Typical noise level        : {diagnostics['noise_level']:.10f}")

print("\nSmart recommendation:")
print(f"- Main method     : {recommendation['method']}")
print(f"- Reason          : {recommendation['reason']}")
print(f"- Adopted period  : {adopted_period:.10f} days = {format_period(adopted_period, unit_info)}")
if np.isfinite(relative_spread):
    print(f"- Method spread   : {relative_spread:.6f}")

print("\nSaved files:")
for file_path in current_outputs:
    print(f"- {os.path.relpath(file_path, SCRIPT_DIR)}")
