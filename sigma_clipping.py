import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

# =========================================================
# SETTINGS
# =========================================================
FILENAME = "APTEST.csv"   # <-- сюда впиши имя своего файла
SIGMA = 3.0

# =========================================================
# LOAD FILE
# =========================================================
try:
    df = pd.read_csv(FILENAME, comment="#")
except FileNotFoundError:
    raise FileNotFoundError(
        f"File '{FILENAME}' not found. Put the CSV file in the same folder as this script "
        f"or specify the full path."
    )

print("Columns found:", list(df.columns))

# =========================================================
# AUTO COLUMN DETECTION
# =========================================================
def find_column(columns, keywords):
    """
    Search for the first column whose name contains one of the keywords.
    """
    lower_columns = [str(col).strip() for col in columns]

    for key in keywords:
        key = key.lower()
        for col in lower_columns:
            if key in col.lower():
                return col
    return None


time_col = find_column(df.columns, ["jd", "hjd", "mjd", "bjd", "time"])
mag_col  = find_column(df.columns, ["mag", "magnitude"])
flux_col = find_column(df.columns, ["flux", "intensity"])

if time_col is None:
    raise ValueError(
        "No time column found. Expected something like JD, HJD, MJD, BJD, or TIME."
    )

if mag_col is None and flux_col is None:
    raise ValueError(
        "No brightness column found. Expected MAG/MAGNITUDE or FLUX."
    )

print(f"Detected time column: {time_col}")
if mag_col is not None:
    print(f"Detected magnitude column: {mag_col}")
else:
    print(f"Detected flux column: {flux_col}")

# =========================================================
# PREPARE DATA
# =========================================================
work = df.copy()

# Convert selected columns to numeric
work[time_col] = pd.to_numeric(work[time_col], errors="coerce")

if mag_col is not None:
    work[mag_col] = pd.to_numeric(work[mag_col], errors="coerce")
else:
    work[flux_col] = pd.to_numeric(work[flux_col], errors="coerce")

# If magnitude is absent but flux exists, convert flux -> magnitude
if mag_col is None:
    work = work.dropna(subset=[time_col, flux_col]).copy()
    work = work[work[flux_col] > 0].copy()

    if len(work) == 0:
        raise ValueError("No positive flux values left after filtering.")

    work["Mag"] = -2.5 * np.log10(work[flux_col])
    mag_col = "Mag"

# Rename to standard names
work = work.rename(columns={time_col: "JD", mag_col: "Mag"})

# Keep only valid numeric rows
work = work.dropna(subset=["JD", "Mag"]).copy()
work = work[np.isfinite(work["JD"]) & np.isfinite(work["Mag"])].copy()

if len(work) == 0:
    raise ValueError("No valid rows left after cleaning NaN / invalid values.")

# Sort by time
work = work.sort_values("JD").reset_index(drop=True)

print(f"Rows after basic cleaning: {len(work)}")

# =========================================================
# SIGMA CLIPPING
# =========================================================
clipped = sigma_clip(
    work["Mag"].values,
    sigma=SIGMA,
    maxiters=None,
    cenfunc="median",
    stdfunc="mad_std"
)

mask_good = ~clipped.mask

df_clean = work.loc[mask_good].copy()
df_out   = work.loc[~mask_good].copy()

print(f"Kept points   : {len(df_clean)}")
print(f"Removed points: {len(df_out)}")

# =========================================================
# SAVE RESULTS
# =========================================================
df_clean.to_csv("clean_data.csv", index=False)
df_out.to_csv("outliers.csv", index=False)

print("Saved files:")
print(" - clean_data.csv")
print(" - outliers.csv")

# =========================================================
# VISUALIZATION
# =========================================================
plt.figure(figsize=(10, 5))

plt.scatter(
    df_clean["JD"],
    df_clean["Mag"],
    s=12,
    label="Clean data"
)

if len(df_out) > 0:
    plt.scatter(
        df_out["JD"],
        df_out["Mag"],
        s=24,
        color="red",
        label="Outliers"
    )

plt.gca().invert_yaxis()
plt.xlabel("JD")
plt.ylabel("Magnitude")
plt.title(f"Sigma clipping ({SIGMA}σ)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
