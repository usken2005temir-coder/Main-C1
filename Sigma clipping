import numpy as np
import pandas as pd
from astropy.stats import sigma_clip

# ========================= LOAD FILE ============================
FILENAME = "your_file.csv"  # <-- любой файл

df = pd.read_csv(FILENAME, comment="#")

# ========================= AUTO COLUMN DETECT ===================
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

# ========================= PREPARE DATA =========================
work = df.copy()

# если только flux → переводим в magnitude
if mag_col is None:
    work = work[work[flux_col] > 0].copy()
    work["Mag"] = -2.5 * np.log10(work[flux_col])
    mag_col = "Mag"

work = work.rename(columns={time_col: "JD", mag_col: "Mag"})
work = work.dropna(subset=["JD", "Mag"])
work = work.sort_values("JD")

# ========================= SIGMA CLIPPING =======================
SIGMA = 3.0

clipped = sigma_clip(
    work["Mag"].values,
    sigma=SIGMA,
    maxiters=None,     # пока не стабилизируется
    cenfunc="median",
    stdfunc="mad_std"
)

mask_good = ~clipped.mask

df_clean = work[mask_good]
df_out   = work[~mask_good]

print(f"Kept: {len(df_clean)}")
print(f"Removed: {len(df_out)}")

# ========================= SAVE ================================
df_clean.to_csv("clean_data.csv", index=False)
df_out.to_csv("outliers.csv", index=False)

print("Saved: clean_data.csv, outliers.csv")

# ========================= VISUALIZATION ========================
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.scatter(df_clean["JD"], df_clean["Mag"], s=10, label="Clean")
plt.scatter(df_out["JD"], df_out["Mag"], color="red", s=20, label="Outliers")
plt.gca().invert_yaxis()
plt.legend()
plt.grid()
plt.title("Sigma clipping (3σ)")
plt.show()
