# Comparative Study of Data Generation Techniques (Data‑level) for Imbalanced Regression

This project is part of a Master's degree thesis made by António Pedro Pinheiro and supervised by Rita P. Ribeiro.
  - Thesis URL: <https://hdl.handle.net/10216/169022>
  - Code Ocean Reproducible Capsule URL: <https://doi.org/10.24433/CO.7826905.v2>


This repository presents a comprehensive **comparative study** of classical and novel **tabular data generation techniques** for **imbalanced regression** tasks, including:

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
├── script_interdependencies.jpg                            # Diagram displaying the Python files interdependencies
├── CARTGen-IR_algorithm.jpg                                # Flowchart outlining the originally proposed CARTGen-IR algorithm steps
├── results/                                                # Outputs: results, logs, and charts
└── requirements.txt                                        # Python dependencies
```

---

## How to Use

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the experiments**:
   ```bash
   python automated_script_datasets_final.py
   ```
   This will:
   - Load all datasets from `datasets/`
   - Compute relevance using `phi(y)`
   - Execute a full **stratified repeated 2×5-fold CV pipeline** across augmentation methods and regressors
   - Save detailed metrics in `results/all_results_df.csv` and summaries in `results/summary_df.csv`

---

## Generation Techniques

Implemented methods include:

- **Undersampling/Oversampling**: RU, RO, WERCS  
- **Introduction of Noise**: Gaussian Noise (GN)  
- **SMOTE Variants**: SMOTER, SMOGN, WSMOTER, G-SMOTER  
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

Final results provide **mean ± std** across **10 folds**, within a stratified repeated 2 x 5-Fold Cross-Validation Pipeline.

---

## Results

Find the outputs in `results/`:

- `all_results_df.csv`: fold-level metrics per dataset, strategy, parameters, and model  
- `summary_df.csv`: aggregated results (mean ± std)  
- Plots and Tables generated for the analysis of the results, namely Wilcoxon Signed-Rank Test, Bayesian Signed Rank Test, Friedman Test, Nemenyi Test + Critical Difference Diagrams and Runtime Statistics

---

## Background & References

This work leverages several contributions in the literature:

- Proposed data-level strategies for IR
- Regression adaptations of **SMOTER**  
- Relevance-based sampling
- Synthetic data generation via **CART**  
- Relevance Weighted error metrics for IR: **RW.MSE, RW-RMSE, RW-MAE, RW-R², SERA**

---

## License & Acknowledgments

- **License**: MIT (see `LICENSE`)  

---
