import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = "clean_data.csv"   # файл после sigma clipping





frequency, power = ls.autopower(
)

# BEST PERIOD
best_period = 1.0 / best_frequency




plt.title("Lomb-Scargle Periodogram")
plt.legend()
plt.tight_layout()
plt.show()


plt.xlabel("Phase")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
plt.show()


