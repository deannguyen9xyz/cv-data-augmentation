import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

#=================
# 1. Load EIS data
#=================
scripts_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(scripts_dir)
data_path = os.path.join(base_dir, "data", "CV_data.csv")

df = pd.read_csv(data_path, header=0, skiprows=[1])

#====================================
# 2. CV Augmentation (Baseline Drift)
#====================================
def varry_CV(CV_data):
    voltage = CV_data['Voltage'].values
    current = CV_data['Current'].values

    # Simulate baseline drift
    # Random linear drift coefficient (±5% of current range)
    drift_coef = np.random.uniform(-0.05, 0.05) * (max(current) - min(current))
    
    # Random quadratic drift coefficient (smaller effect)
    quad_coef = np.random.uniform(-0.02, 0.02) * (max(current) - min(current))
    
    # Generate drift: linear + quadratic term
    drift = drift_coef * voltage + quad_coef * voltage**2

    # Add drift to current
    drifted_current = current + drift

    # Return modified DataFrame
    new_CV = pd.DataFrame({'Voltage': voltage, 'Current': drifted_current})

    return new_CV

#====================================
# 3. Generate n-1 augmented CV datasets
#====================================
results_dir = os.path.join(base_dir, "results","script2")

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