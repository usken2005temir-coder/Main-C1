from astropy.timeseries import LombScargle
# ============================================================
# LOMB-SCARGLE PERIOD ANALYSIS
# ============================================================

# use cleaned data
t = df_clean["JD"].values
y = df_clean["Mag"].values

# optional normalization: remove mean magnitude
y_mean = np.mean(y)
y_centered = y - y_mean

# ------------------------------------------------------------
# FREQUENCY RANGE
# ------------------------------------------------------------
# You can adjust these values depending on expected periods
# frequency is in 1/day if JD is in days
MIN_FREQUENCY = 0.01   # cycles/day
MAX_FREQUENCY = 20.0   # cycles/day

# create Lomb-Scargle object
ls = LombScargle(t, y_centered)

# compute periodogram
frequency, power = ls.autopower(
    minimum_frequency=MIN_FREQUENCY,
    maximum_frequency=MAX_FREQUENCY,
    samples_per_peak=10
)

# ------------------------------------------------------------
# BEST PERIOD
# ------------------------------------------------------------
best_frequency = frequency[np.argmax(power)]
best_period = 1.0 / best_frequency
best_power = power.max()

print("\n===== LOMB-SCARGLE RESULTS =====")
print(f"Best frequency: {best_frequency:.8f} cycles/day")
print(f"Best period:    {best_period:.8f} days")
print(f"Max power:      {best_power:.6f}")

# optional false alarm probability
try:
    fap = ls.false_alarm_probability(best_power)
    print(f"False alarm probability: {fap:.6e}")
except Exception as e:
    print("False alarm probability could not be computed:", e)

# save periodogram
periodogram_df = pd.DataFrame({
    "frequency": frequency,
    "period": 1.0 / frequency,
    "power": power
})
periodogram_df.to_csv("lomb_scargle_periodogram.csv", index=False)
print("Saved: lomb_scargle_periodogram.csv")

# ------------------------------------------------------------
# PLOT PERIODROGRAM
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(frequency, power, linewidth=1.2)
plt.axvline(best_frequency, linestyle="--", label=f"Best f = {best_frequency:.5f}")
plt.xlabel("Frequency [cycles/day]")
plt.ylabel("Power")
plt.title("Lomb-Scargle Periodogram")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lomb_scargle_periodogram.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# PHASE FOLDING
# ------------------------------------------------------------
phase = (t / best_period) % 1.0

# sort by phase for nicer plotting
sort_idx = np.argsort(phase)
phase_sorted = phase[sort_idx]
mag_sorted = y[sort_idx]

# model on smooth phase grid
phase_fit = np.linspace(0, 1, 1000)
t_fit = phase_fit * best_period
model_mag = ls.model(t_fit, best_frequency) + y_mean

# ------------------------------------------------------------
# PLOT PHASED LIGHT CURVE
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(phase_sorted, mag_sorted, s=18, label="Clean data")
plt.scatter(phase_sorted + 1.0, mag_sorted, s=18, label="Repeated phase")
plt.plot(phase_fit, model_mag, linewidth=2, label="LS model")
plt.plot(phase_fit + 1.0, model_mag, linewidth=2)

plt.gca().invert_yaxis()
plt.xlabel("Phase")
plt.ylabel("Magnitude")
plt.title(f"Phase-folded Light Curve (P = {best_period:.6f} d)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("phase_folded_ls.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# SAVE PHASE DATA
# ------------------------------------------------------------
phase_df = pd.DataFrame({
    "JD": t,
    "phase": phase,
    "Mag": y
}).sort_values("phase")

phase_df.to_csv("phase_folded_data.csv", index=False)
print("Saved: phase_folded_data.csv")
print("Saved: lomb_scargle_periodogram.png")
print("Saved: phase_folded_ls.png")
