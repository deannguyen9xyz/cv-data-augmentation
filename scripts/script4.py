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

#=============================================
# 2. CV Augmentation (Gaussian Noise Addition)
#=============================================
def varry_CV(CV_data):
    voltage = CV_data['Voltage'].values
    current = CV_data['Current'].values

    # ----------------------
    # 1. Add Gaussian noise to current
    # ----------------------
    noise_level = np.random.uniform(0.0005, 0.001) * (max(current) - min(current))  # 1-5% of current range
    noisy_current = current + np.random.normal(0, noise_level, size=current.shape)

    # ----------------------
    # 2. Add small Gaussian noise to voltage (optional wrapping effect)
    # ----------------------
    voltage_noise_level = np.random.uniform(0.0005, 0.001) * (max(voltage) - min(voltage))
    noisy_voltage = voltage + np.random.normal(0, voltage_noise_level, size=voltage.shape)

    # Return modified DataFrame
    new_CV = pd.DataFrame({'Voltage': noisy_voltage, 'Current': noisy_current})

    return new_CV

#====================================
# 3. Generate n-1 augmented CV datasets
#====================================
results_dir = os.path.join(base_dir, "results","script4")

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