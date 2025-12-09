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

#==============================
# 2. CV Augmentation (combined)
#==============================
def varry_CV(CV_data):
    voltage = CV_data['Voltage'].values
    current = CV_data['Current'].values

    # ----------------------
    # 1. Voltage Shift + Current Scaling
    # ----------------------

    # Small random voltage shift (±20 mV)
    shift_amount = np.random.uniform(-0.02, 0.02)
    aug_voltage = voltage + shift_amount

    # Random current scaling (±10%)
    current_scale_factor = np.random.uniform(0.9, 1.1)
    aug_current = current * current_scale_factor

    # ----------------------
    # 2. Baseline Drift
    # ----------------------

    # Random linear drift coefficient (±5% of current range)
    drift_coef = np.random.uniform(-0.05, 0.05) * (max(aug_current) - min(aug_current))
    
    # Random quadratic drift coefficient (smaller effect)
    quad_coef = np.random.uniform(-0.02, 0.02) * (max(aug_current) - min(aug_current))
    
    # Generate drift: linear + quadratic term
    drift = drift_coef * aug_voltage + quad_coef * aug_voltage**2

    # Add drift to current
    aug_current = aug_current + drift

    # ----------------------
    # 3. Peak Broadening
    # ----------------------
    # Randomly choose a broadening factor (sigma for Gaussian smoothing)
    sigma = np.random.uniform(1, 5)  # Adjust for more/less smoothing
    aug_current = gaussian_filter1d(aug_current, sigma=sigma)

    # ----------------------
    # 4. Voltage Wrapping
    # ----------------------
    # Simulate small shifts in voltage (wrap effect)
    wrap_factor = np.random.uniform(-0.02, 0.02) * (max(aug_voltage) - min(aug_voltage))
    wrapped_voltage = aug_voltage + wrap_factor * np.sin(2 * np.pi * (aug_voltage - min(aug_voltage)) / (max(aug_voltage) - min(aug_voltage)))
    
    # ----------------------
    # 5. Add Gaussian noise to current
    # ----------------------
    noise_level = np.random.uniform(0.0001, 0.0005) * (max(aug_current) - min(aug_current))  # 1-5% of current range
    aug_current = aug_current + np.random.normal(0, noise_level, size=aug_current.shape)

    # ----------------------
    # 6. Add small Gaussian noise to voltage (optional wrapping effect)
    # ----------------------
    voltage_noise_level = np.random.uniform(0.0001, 0.0005) * (max(aug_voltage) - min(aug_voltage))
    aug_voltage = aug_voltage + np.random.normal(0, voltage_noise_level, size=aug_voltage.shape)

    # Return modified DataFrame
    new_CV = pd.DataFrame({'Voltage': aug_voltage,'Current': aug_current})

    return new_CV

#====================================
# 3. Generate n-1 augmented CV datasets
#====================================
results_dir = os.path.join(base_dir, "results","script5")

# Create results folder if not exists
os.makedirs(results_dir, exist_ok=True)

n=11

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