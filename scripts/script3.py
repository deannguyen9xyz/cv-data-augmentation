import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import random
import os

#=================
# 1. Load EIS data
#=================
scripts_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(scripts_dir)
data_path = os.path.join(base_dir, "data", "CV_data.csv")

df = pd.read_csv(data_path, header=0, skiprows=[1])

#========================================================
# 2. CV Augmentation (Peak Broadening + Voltage Wrapping)
#========================================================
def varry_CV(CV_data):
    voltage = CV_data['Voltage'].values
    current = CV_data['Current'].values

    # ----------------------
    # 1. Peak Broadening
    # ----------------------
    # Randomly choose a broadening factor (sigma for Gaussian smoothing)
    sigma = np.random.uniform(1, 5)  # Adjust for more/less smoothing
    broadened_current = gaussian_filter1d(current, sigma=sigma)

    # ----------------------
    # 2. Voltage Wrapping
    # ----------------------
    # Simulate small shifts in voltage (wrap effect)
    wrap_factor = np.random.uniform(-0.02, 0.02) * (max(voltage) - min(voltage))
    wrapped_voltage = voltage + wrap_factor * np.sin(2 * np.pi * (voltage - min(voltage)) / (max(voltage) - min(voltage)))

    # Return modified DataFrame
    new_CV = pd.DataFrame({'Voltage': wrapped_voltage, 'Current': broadened_current})

    return new_CV

#====================================
# 3. Generate n-1 augmented CV datasets
#====================================
results_dir = os.path.join(base_dir, "results","script3")

# Create results folder if not exists
os.makedirs(results_dir, exist_ok=True)

n=6

augmented_files = []
for i in range(1, n):
    augmented = varry_CV(df)

    output_path = os.path.join(results_dir, f"cv_augmented_{i}.csv")
    augmented.to_csv(output_path, index=False)

    augmented_files.append(pd.read_csv(output_path))

    print(f"Saved: {output_path}")

print("Done! augmented CV files created.")

#===================
# 4. Plot all curves
#===================

plt.figure(figsize=(8, 6))

# Plot raw CV
plt.plot(df["Voltage"], df["Current"], label="Raw CV", linewidth=3, linestyle='--', color='black')

# Plot augmented CVs
for i, aug_df in enumerate(augmented_files, start=1):
    plt.plot(aug_df["Voltage"], aug_df["Current"], label=f"Augmented {i}")

# Graph styling
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("Raw vs Augmented Cyclic Voltammograms")
plt.legend()
plt.grid(True)

plt.show()