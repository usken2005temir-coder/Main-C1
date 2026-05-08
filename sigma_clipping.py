import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

# ============================================================
# DEFAULT SETTINGS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = None
OUTPUT_DIR = None

SIGMA = 3.0
WINDOW = 15
SHOW_PLOTS = True

# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sigma clipping for variable-star light curves."
    )

    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)

    parser.add_argument("--time-col", default=None)
    parser.add_argument("--signal-col", default=None)
    parser.add_argument("--error-col", default=None)

    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--window", type=int, default=WINDOW)

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


def display_label_for_signal(column_name, converted_from_flux=False):
    if converted_from_flux:
        return "Magnitude"

    name = str(column_name)
    normalized = normalize_name(name)

    if "mag" in normalized or "magnitude" in normalized:
        return "Magnitude"

    if normalized == "vr":
        return "V/R"

    if normalized in {"rv", "radialvelocity", "velocity"}:
        return "RV"

    return name


def is_magnitude_label(label):
    normalized = normalize_name(label)
    return "mag" in normalized or "magnitude" in normalized


def choose_columns(df, args):
    columns = list(df.columns)

    time_col = args.time_col or find_column(
        columns,
        ["JD", "HJD", "BJD", "MJD", "time", "date"],
        forbidden=["error", "err", "sigma"]
    )

    signal_col = args.signal_col or find_column(
        columns,
        [
            "Mag", "Magnitude", "Vmag", "gmag", "rmag", "imag",
            "V/R", "VR", "RV", "Radial Velocity", "Velocity",
            "Flux", "brightness", "Signal", "Value", "V", "R", "I",
        ],
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
                [
                    "Mag Error", "mag_err", "magerr", "e_mag", "merr",
                    "RV Error", "rv_err", "rverr", "e_rv", "velocity_error", "vel_err",
                    "Signal Error", "signal_err", "dy", "yerr",
                ]
            )

    return time_col, signal_col, err_col

# ============================================================
# PATH HELPERS
# ============================================================
def supported_input_file(path):
    extension = os.path.splitext(path)[1].lower()
    return extension in {".csv", ".txt", ".tsv", ".dat"}


def find_default_input():
    aptest_dirs = [
        os.path.join(SCRIPT_DIR, "APTEST"),
        os.path.join(os.getcwd(), "APTEST"),
    ]

    for aptest_dir in aptest_dirs:
        if not os.path.isdir(aptest_dir):
            continue

        files = [
            os.path.join(aptest_dir, name)
            for name in os.listdir(aptest_dir)
            if os.path.isfile(os.path.join(aptest_dir, name))
            and supported_input_file(os.path.join(aptest_dir, name))
        ]

        if len(files) == 1:
            return os.path.abspath(files[0])

        if len(files) > 1:
            file_names = [os.path.basename(path) for path in files]
            raise FileExistsError(
                "APTEST must contain exactly one input data file when --input is not used. "
                f"Found {len(files)} files: {file_names}. "
                "Leave only one file in APTEST or pass --input explicitly."
            )

    raise FileNotFoundError(
        "Could not find an input table in APTEST. "
        "Put exactly one .csv, .txt, .tsv or .dat file into APTEST, "
        "or pass --input path/to/file."
    )


def infer_output_dir(input_path, output_dir):
    if output_dir is not None:
        return os.path.abspath(output_dir)

    input_dir = os.path.dirname(os.path.abspath(input_path))

    if os.path.basename(input_dir).lower() == "aptest":
        return os.path.dirname(input_dir)

    return input_dir


# ============================================================
# TABLE READING
# ============================================================
def is_float_like(value):
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


def infer_headerless_signal_name(path):
    file_name = normalize_name(os.path.splitext(os.path.basename(path))[0])

    if "vr" in file_name:
        return "V/R"

    if "rv" in file_name or "radialvelocity" in file_name:
        return "RV"

    if "flux" in file_name:
        return "Flux"

    if "mag" in file_name or "phot" in file_name:
        return "Mag"

    return "Signal"


def assign_headerless_columns(table, path):
    n_columns = len(table.columns)
    signal_name = infer_headerless_signal_name(path)

    if n_columns == 2:
        table.columns = ["JD", signal_name]
    elif n_columns == 3:
        table.columns = ["JD", signal_name, f"{signal_name} Error"]
    else:
        table.columns = [f"col_{index + 1}" for index in range(n_columns)]

    return table


def read_headerless_table(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".csv":
        table = pd.read_csv(path, comment="#", header=None)
    elif extension == ".tsv":
        table = pd.read_csv(path, comment="#", sep="\t", header=None)
    else:
        table = pd.read_csv(path, comment="#", sep=r"\s+", engine="python", header=None)

    return assign_headerless_columns(table, path)


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

    if len(table.columns) >= 2 and all(is_float_like(column) for column in table.columns):
        table = read_headerless_table(path)

    return table

# ============================================================
# LOAD FILE
# ============================================================
args = parse_args()

if args.sigma <= 0:
    raise ValueError("sigma must be positive.")

if args.window < 3:
    raise ValueError("window must be at least 3.")

input_file = os.path.abspath(args.input) if args.input is not None else find_default_input()
output_dir = infer_output_dir(input_file, args.output_dir)

df = read_input_table(input_file)

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
# PREPARE DATA
# ============================================================
work = df.copy()

work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
work[signal_col] = pd.to_numeric(work[signal_col], errors="coerce")

if err_col is not None:
    work[err_col] = pd.to_numeric(work[err_col], errors="coerce")

signal_name = normalize_name(signal_col)
signal_label = display_label_for_signal(signal_col, converted_from_flux=("flux" in signal_name))

if "flux" in signal_name:
    valid_flux = np.isfinite(work[signal_col]) & (work[signal_col] > 0)
    work = work.loc[valid_flux].copy()

    flux = work[signal_col].to_numpy()
    work["Mag"] = -2.5 * np.log10(flux)

    if err_col is not None:
        flux_error = work[err_col].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            work["Mag Error"] = 2.5 / np.log(10.0) * flux_error / flux

    mag_col = "Mag"
else:
    mag_col = signal_col

rename_map = {time_col: "JD"}
if mag_col != "Mag":
    rename_map[mag_col] = "Mag"

work = work.rename(columns=rename_map)
work["Signal Label"] = signal_label
work["Original Signal Column"] = signal_col
work = work.dropna(subset=["JD", "Mag"]).copy()
work = work.sort_values("JD").reset_index(drop=True)

if len(work) < 5:
    raise ValueError("Too few valid data points for sigma clipping.")

# ============================================================
# BUILD LOCAL TREND
# ============================================================
work["Trend"] = work["Mag"].rolling(
    window=args.window,
    center=True,
    min_periods=1
).median()

work["Residual"] = work["Mag"] - work["Trend"]

# ============================================================
# SIGMA CLIPPING
# ============================================================
clipped = sigma_clip(
    work["Residual"].to_numpy(),
    sigma=args.sigma,
    maxiters=None,
    cenfunc="median",
    stdfunc="mad_std"
)

mask_good = ~clipped.mask

df_clean = work.loc[mask_good].copy()
df_out = work.loc[~mask_good].copy()

# ============================================================
# SAVE RESULTS
# ============================================================
os.makedirs(output_dir, exist_ok=True)

output_clean = os.path.join(output_dir, "clean_data.csv")
output_outliers = os.path.join(output_dir, "outliers.csv")
output_plot = os.path.join(output_dir, "sigma_clipping_result.png")

df_clean.to_csv(output_clean, index=False)
df_out.to_csv(output_outliers, index=False)

print("=" * 70)
print("Sigma clipping results")
print(f"Input file                : {input_file}")
print(f"Output directory          : {output_dir}")
print(f"Sigma threshold           : {args.sigma}")
print(f"Rolling median window     : {args.window} points")
print(f"Number of points before   : {len(work)}")
print(f"Number of clean points    : {len(df_clean)}")
print(f"Number of outliers        : {len(df_out)}")
print("=" * 70)

# ============================================================
# VISUALIZATION
# ============================================================
plt.figure(figsize=(12, 6))

plt.scatter(df_clean["JD"], df_clean["Mag"], s=18, label="Clean data")
plt.scatter(df_out["JD"], df_out["Mag"], s=30, color="red", label="Outliers")
plt.plot(work["JD"], work["Trend"], linewidth=1.5, label="Rolling median")

if is_magnitude_label(signal_label):
    plt.gca().invert_yaxis()
plt.xlabel("JD")
plt.ylabel(signal_label)
plt.title(f"Sigma clipping on residuals ({args.sigma:g}-sigma)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)

if args.show_plots:
    plt.show()

plt.close()

# ============================================================
# FINAL MESSAGE
# ============================================================
print("\nSaved files:")
print(f"- {output_clean}")
print(f"- {output_outliers}")
print(f"- {output_plot}")
