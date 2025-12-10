# cv-data-augmentation
Scripts for augmenting cyclic voltammetry datasets to improve machine-learning robustness.

## 🌟 Key Features

This project provides various Python scripts to apply common data augmentation techniques to Cyclic Voltammetry (CV) data. Each script loads the raw CV data, applies a specific augmentation method, saves the augmented data, and generates a comparison plot.

---

## ⚙️ Requirements

Place the input CSV data file named **"CV_data.csv"** in the **"data"** directory alongside the Python script(s).

Install required libraries using:

```
pip install numpy scipy pandas matplotlib
```

---

## 📁 Project Files

| File Name | Location | Description |
| :--- | :--- | :--- |
| `CV_data.csv` | `data/` | **Raw cyclic voltammetry data** for input. |
| `script1.py` | `scripts/` | Augmentation using **Simple Voltage Shift** and **Current Scaling**. |
| `script2.py` | `scripts/` | Augmentation using **Baseline Drift**. |
| `script3.py` | `scripts/` | Augmentation using **Peak Broadening** and **Voltage Wrapping**. |
| `script4.py` | `scripts/` | Augmentation using **Gaussian Noise Addition**. |
| `script5.py` | `scripts/` | Augmentation combining **all previous techniques**. |

---

## 📊 Input Data Format

Each script expects a **CSV file** with the following columns:

```
Voltage, Current
V, A
-0.1, -2.30E-05
-0.11, -2.40E-05
...
```

* `Potential` → Applied potential (Voltage, V)
* `Current` → Measured current (Amperes, A)

---
## ▶️ How to Run

Run each script separately.

---

## 📈 Output

Each script generates:

* ✅ Augmented CSV data saved to the corresponding folder in the *results/* directory.
* ✅ A comparison plot (raw vs. augmented CV) saved to the *figures/* directory.

---

## 🎯 Purpose of This Project

This project is designed for:

* Demonstrating common CV data augmentation techniques
* Creating diverse datasets for training machine learning models in electrochemistry
* GitHub portfolio demonstration

---

## 📌 Future Improvements

* Implement more advanced augmentation techniques.

--- 

## 🧑‍💻 Author

Developed by: Vu Bao Chau Nguyen, Ph.D.
Keywords: Cyclic Voltammetry (CV), Data Augmentation, Machine Learning, Electrochemistry.
