# SyntheticTabularDataAugmentation‑IR

**Comparative Study of Data Augmentation Techniques (Data‑level) for Imbalanced Regression**

A Master's degree thesis project by **António Pedro Pinheiro**, supervised by **Rita P. Ribeiro**.

This repository presents a comprehensive **comparative study** of classical and novel **tabular data augmentation techniques** for **imbalanced regression** tasks, including:

- A wide range of **existing literature methods**:
  - Undersampling/oversampling (RU, RO, WERCS)
  - Introduction of Noise (Gaussian Noise - GN)
  - SMOTE-based techniques adapted for regression (SMOTER, SMOGN, WSMOTER, G-SMOTER)
  - Deep Learning (DAVID)
  - Regression-specific kNN (KNNOR-REG)
  - A **CART-based custom generator**: **CARTGen‑IR**

---

## Repository Structure

```
.
├── functions/                                              # Core augmentation routines, utilities and functions
├── datasets/                                               # Collection of CSV datasets
├── automated_script_datasets_final.py                      # Main experiment pipeline script
├── automated_script_datasets_final_with_XGBoost_SERA.py    # Main experiment pipeline script withy an addotional learning model: XGboost with a custom objective function based on SERA
├── results/                                                # Outputs: results, logs, and charts
└── requirements.txt                                        # Python dependencies
```

---

## Getting Started

1. **Clone** the repository:
   ```bash
   git clone https://github.com/antoniopedropi/SyntheticTabularDataAugmentation-IR.git
   cd SyntheticTabularDataAugmentation-IR
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the experiments**:
   ```bash
   python automated_script_datasets_final.py
   ```
   This will:
   - Load all datasets from `datasets/`
   - Compute relevance using `phi(y)`
   - Execute a full **stratified repeated 2×5-fold CV pipeline** across augmentation methods and regressors
   - Save detailed metrics in `results/all_results_df.csv` and summaries in `results/summary_df.csv`

---

## Augmentation Techniques

Implemented methods include:

- **Undersampling/Oversampling**: RU, RO, WERCS  
- **Introduction of Noise**: Gaussian Noise (GN)  
- **SMOTE Variants**: SMOTER, SMOGN, WSMOTER, GSMOTER  
- **Deep Learning**: DAVID  
- **Regression-specific kNN**: KNNOR‑REG  
- **CART-based Synthesizer**: *CARTGen‑IR* (original)

---

## Regressors & Evaluation

Models tested:

- **RandomForestRegressor (RF)**
- **SupportVectorRegressor (SVM / SVR)**
- **XGBoost Regression (XGB)**

Metrics captured per fold:

```
MSE, RMSE, MAE, R²,
Weighted variants: RW-MSE, RW-RMSE, RW-MAE, RW-R², DW-RMSE (denseweight weighted root mean quared error),
+ SERA (area under relevance-weighted error) + DW-SERA (denseweight weighted SERA)
```

Final results provide **mean ± std** across **10 folds**, with a stratified repeated 2 x 5-Fold Cross Validation Pipeline.

---

## 🛠️ Custom CARTGen‑IR Synthesizer

The **CARTGen‑IR** method uses a **rarity-weighted CART model** to generate new data:

- Learns regression trees weighted by target rarity
- Generates synthetic instances along tree leaves
- Integrates into CV pipeline as augmentation strategy `"CARTGen-IR"`

---

## Results

Find the outputs in `results/`:

- `all_results_df.csv`: fold-level metrics per dataset, strategy, parameters, and model  
- `summary_df.csv`: aggregated results (mean ± std)  
- Plots (e.g., boxplots, relevance curves) generated alongside per-dataset or aggregated stats

---

## Background & References

This work leverages several contributions:

- Regression adaptations of **SMOTER/SMOGN**  
- Relevance-based sampling (*phi*-based) 
- Synthetic data generation via **CART**  
- Weighted error metrics: **WMSE, WMAE, WR²**, **SERA**

---

## License & Acknowledgments

- **Author**: António Pedro Pinheiro (Master’s thesis)  
- **Supervisor**: Rita P. Ribeiro  
- **License**: MIT (see `LICENSE`)  
- Based on MSc research at Faculty of Science on University of Porto.

---
