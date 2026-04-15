import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

# ============================================================
# LOAD FILE
# ============================================================
FILENAME = "APTEST/APTEST.csv"
df = pd.read_csv(FILENAME, comment="#")


# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================
def find_column(columns, names):
    for name in names:
        for col in columns:
            if name.lower() in col.lower():
                return col
    return None


time_col = find_column(df.columns, ["jd", "hjd", "mjd", "time"])
mag_col  = find_column(df.columns, ["mag", "magnitude"])
flux_col = find_column(df.columns, ["flux"])

if time_col is None:
    raise ValueError("No time column found")

if mag_col is None and flux_col is None:
    raise ValueError("No MAG or FLUX column found")


# ============================================================
# PREPARE DATA
# ============================================================
work = df.copy()

# if only FLUX is available, convert it to magnitude
if mag_col is None:
    work = work[work[flux_col] > 0].copy()
    work["Mag"] = -2.5 * np.log10(work[flux_col])
    mag_col = "Mag"

work = work.rename(columns={time_col: "JD", mag_col: "Mag"})
work = work.dropna(subset=["JD", "Mag"]).copy()
work = work.sort_values("JD").reset_index(drop=True)


# ============================================================
# CHECK RAW DATA
# ============================================================
print("Number of points before clipping:", len(work))
print("Time column:", "JD")
print("Signal column:", "Mag")


# ============================================================
# BUILD LOCAL TREND
# ============================================================
WINDOW = 15
work["Trend"] = work["Mag"].rolling(
    window=WINDOW,
    center=True,
    min_periods=1
).median()

work["Residual"] = work["Mag"] - work["Trend"]


# ============================================================
# SIGMA CLIPPING
# ============================================================
SIGMA = 3.0

clipped = sigma_clip(
    work["Residual"].values,
    sigma=SIGMA,
    maxiters=None,
    cenfunc="median",
    stdfunc="mad_std"
)

mask_good = ~clipped.mask
df_clean = work.loc[mask_good].copy()
df_out   = work.loc[~mask_good].copy()

print("Number of clean points:", len(df_clean))
print("Number of outliers:", len(df_out))


# ============================================================
# SAVE RESULTS
# ============================================================
df_clean.to_csv("clean_data.csv", index=False)
df_out.to_csv("outliers.csv", index=False)

print("Saved: clean_data.csv")
print("Saved: outliers.csv")


# ============================================================
# VISUALIZATION
# ============================================================
plt.figure(figsize=(12, 6))

plt.scatter(df_clean["JD"], df_clean["Mag"], s=18, label="Clean data")
plt.scatter(df_out["JD"], df_out["Mag"], s=30, color="red", label="Outliers")
plt.plot(work["JD"], work["Trend"], linewidth=1.5, label="Rolling median")

plt.gca().invert_yaxis()
plt.xlabel("JD")
plt.ylabel("Magnitude")
plt.title("Sigma clipping on residuals (3σ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
