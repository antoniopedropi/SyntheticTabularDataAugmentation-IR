# SyntheticTabularDataAugmentation‑IR

**Comparative Study of Data Augmentation Techniques (Data‑level) for Imbalanced Regression**

A Master's degree thesis project by **António Pedro Pinheiro**, supervised by **Rita P. Ribeiro**.

This repository presents a comprehensive **comparative study** of classical and novel **tabular data augmentation techniques** for **imbalanced regression** tasks, including:

- A wide range of **existing literature methods**:
  - Undersampling/Oversampling (RU, RO, WERCS)
  - Introduction of Noise (Gaussian Noise - GN)
  - SMOTE-based techniques adapted for regression (SMOTER, SMOGN, WSMOTER, G-SMOTER)
  - Deep Learning (DAVID)
  - Other Strategies (KNNOR-REG)
  - A **CART-based custom generator**: **CARTGen‑IR**

---

## Repository Structure

```
.
├── functions/                                              # Core augmentation functions
├── datasets/                                               # Collection of CSV datasets
├── automated_script_datasets_final.py                      # Main experiment pipeline script
├── automated_script_datasets_final_with_XGBoost_SERA.py    # Main experiment pipeline script with an additional learning model: XGboost with a custom objective function based on SERA
├── results/                                                # Outputs: results, logs, and charts
└── requirements.txt                                        # Python dependencies
```

---

## How to Use

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
- **Other Strategies**: KNNOR‑REG  
- **CART-based Synthesizer**: *CARTGen‑IR* (original)

---

## Regressors & Evaluation

Models tested:

- **Random Forest Regressor (RF)**
- **Support Vector Regressor (SVM / SVR)**
- **XGBoost Regressor (XGB)**

Metrics captured per fold:

- MSE, RMSE, MAE, R²
- Weighted variants: RW-MSE, RW-RMSE, RW-MAE, RW-R², DW-RMSE (denseweight weighted root mean quared error)
- IR specific metrics: SERA (area under relevance-weighted error), DW-SERA (denseweight weighted SERA)
```

Final results provide **mean ± std** across **10 folds**, within a stratified repeated 2 x 5-Fold Cross Validation Pipeline.

---

## Results

Find the outputs in `results/`:

- `all_results_df.csv`: fold-level metrics per dataset, strategy, parameters, and model  
- `summary_df.csv`: aggregated results (mean ± std)  
- Plots and Tables generated for the analysis of the results, namely Wilcoxon Signed-Rank Test, Bayesian Signed Rank Test, Friedman Test, Nemenyi Test + Critical Difference Diagrams

---

## Background & References

This work leverages several contributions in the literature:

- Proposed data-level strategies for IR
- Regression adaptations of **SMOTER**  
- Relevance-based sampling
- Synthetic data generation via **CART**  
- Weighted error metrics for IR: **WMSE, WRMSE, WMAE, WR², SERA**

---

## License & Acknowledgments

- **Author**: António Pedro Pinheiro (Master’s thesis)  
- **Supervisor**: Rita P. Ribeiro  
- **License**: MIT (see `LICENSE`)  
- Based on MSc research at Faculty of Science of the University of Porto.

---
