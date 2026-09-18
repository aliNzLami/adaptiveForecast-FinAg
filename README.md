# Adaptive Model Selection for Corn Futures

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Submitted_to_Agricultural_and_Food_Economics-brightgreen.svg)]()

This repository contains the official implementation and replication code for the research paper:

> **"Adaptive Model Selection in Agricultural Finance: A Training-Free Framework for Forecasting in Corn Futures Markets"**  
> *Agricultural and Food Economics Journal*
> 
> **ORCID: 0009-0005-0811-8091**

Code and data for a paper on training-free model selection in agricultural commodity markets. The framework picks forecasting models by matching window statistics to capability profiles. No training. No hyperparameter tuning. Then we look at where the framework switches models, and why.

## Overview

Corn prices swing hard. Droughts, pandemics, wars. A single fixed forecasting model misses these shifts. We test whether a training-free selection framework, first proposed as a theoretical idea, works on real corn futures data.

The pipeline does four things.

1. Calibrates four coefficients once on the full dataset. These set the weight of interpretability, robustness, scalability, and representation capacity.
2. Splits 14 years of daily data into 119 overlapping 90-day windows. Extracts 11 statistical features per window.
3. Ranks five candidate models per window by Manhattan distance between window needs and model profiles. Picks the top two. Evaluates them on RMSE and MAE over the next 30 days.
4. Finds 42 transition points where the framework switches models. Labels each by confidence. Runs SHAP on a separate daily classifier to explain weather drivers.

Key results: the top model beats the second in 68% of windows. Six transitions carry high confidence. All six land on Random Forest. Temperature carries 51.5% of the SHAP mass. Seasonal dummies add 37.7%. Rain sits at 10.0%.

---

## Preprint



---

## Project Structure

``` txt

├── extract_window_features.py             # Sliding window feature extraction
├── transition_points.py                   # Detects and labels transitions
├── interpretability.py                    # SHAP & F1 analysis on daily weather classifier
|
|
|
├── decision-ml-models /
│ └── calibrate.py                          # Coefficient calibration from dataset statistics
│ └── decision_framework_to_windows.py      # Per-window model ranking via Manhattan distance
│ └── final_model_per_window.py             # Picks final model per window
│ ├── model_ranking.py                      # Top-2 selection and RMSE/MAE evaluation
|
│
├── dataset/
│ └── US_Agriculture_Weather_2010_2024.csv  || Real Dataset
│ └── window_features.csv
│ └── model_recommendations.csv
│ └── model_rankings.csv
│ └── top_models_performance.csv
│ └── final_model_per_window.csv
│ └── transition_points.csv
│
|
├── output/
│ ├── micro_analysis_summary.json
│ ├── classification_metrics.csv
│ ├── confusion_matrix.csv
│ ├── shap_global.csv
│ ├── shap_per_class.csv
│ ├── shap_by_group.csv
│ └── test_predictions.csv
│

```
---

## How to Run

Install Python 3.10 or later. Then install dependencies.

```bash
pip install pandas numpy scipy scikit-learn xgboost lightgbm shap
```

Run the pipeline from the repo root.

```bash
python calibrate.py
python extract_window_features.py
python decision_framework_to_windows.py
python model_ranking.py
python final_model_per_window.py
python transition_points.py
python micro_analysis.py
```
Each script prints progress to stdout and writes its outputs. Total runtime is under 15 minutes on a laptop. No GPU needed.
calibrate.py runs a grid search over 104,976 coefficient combinations. That step takes the longest, roughly 8 minutes.

---

## Datasets

This study uses one dataset.

| **Dataset** | **Source** | **Task** | **Key Features / Preprocessing** | **Notes / Purpose** |
| :--- | :--- | :--- | :--- | :--- |
| **US Crop Prices & Midwest Climate Data (2010-2024)** (Invoice-Level) | [Kaggle Link](https://www.kaggle.com/datasets/iconicwasil/us-crop-prices-and-midwest-climate-data-2010-2024?resource=download) | Time-series forecasting of commodity prices; classification of market regimes (Bullish, Bearish, Neutral); feature importance analysis using SHAP & F1 | Daily frequency, 7 columns. Features: Max_Temp_C, Min_Temp_C, Precipitation_mm. Targets: Corn_Price_USD, Soybean_Price_USD, Wheat_Price_USD. Preprocessing: Parsed dates with errors='coerce', computed logarithmic returns, standard kurtosis (fisher=False), and generated 119 overlapping 90-day windows with a 30-day stride. | Primary dataset for all experiments. Merges daily agricultural commodity prices with historical climate data from Iowa (Des Moines), covering January 2010 to January 2024. The dataset is the input for calibration, model ranking, transition detection, and the micro-level SHAP analysis. It provides the raw price and weather series from which all window-level statistical features and daily regime labels are derived. |

---

## Acknowledgements

The author expresses his gratitude to Kaggle for providing the platform that hosts the dataset used in this study, and to Muhammad Wasil for compiling and making the US Crop Prices & Midwest Climate Data (2010-2024) publicly available. The author also thanks GitHub for providing the infrastructure that enabled version control, collaboration, and open-source dissemination of the analysis code and scripts associated with
this research. All computational work was conducted independently, and no external funding or institutional support was received.

---

## License
Code is released under the MIT License. Dataset is subject to the original Kaggle upload's terms. See the Kaggle page for details.

---

## Contact

For questions, issues, or requests regarding the code, please open an issue on this GitHub repository or contact:

- Email: lamiry@financetech.dev
- Researchgate: https://www.researchgate.net/profile/Ali-Nabizade-Lamiry?ev=hdr_xprf
- Website: http://lamiry.netlify.app/

Ali Nabizadeh Lamiry

