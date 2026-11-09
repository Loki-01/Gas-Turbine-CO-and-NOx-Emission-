# Gas Turbine CO and NOx Emissions: Data Analysis and Machine Learning

An end-to-end data science project on hourly sensor data from a gas turbine power plant: data cleaning, exploratory analysis, machine learning, and a careful check of the model errors.

The project looks at three questions:

1. How do the weather and the load of the turbine affect **NOx and CO emissions**?
2. Can the **turbine energy yield (TEY)** be predicted from the sensors, or from the weather alone?
3. Do the models still work on **unseen future years**, and how far can the results be trusted?

> **Main lesson:** the plant changed over time. NOx dropped by about 10 mg/m³ from 2014, so a model trained once on the early years fails on the later years. Retraining regularly on recent data fixes most of the problem.

---

## Results at a glance

All numbers come from a test on the **unseen years 2014-2015**. The models were trained only on 2011-2013.

| Task | Approach | RMSE | MAE | R² |
|---|---|---|---|---|
| TEY from all sensors | Gradient Boosting | 2.75 | 2.49 | **0.966** |
| TEY from weather only | Ridge (best) | 16.00 | 12.87 | -0.14 (no skill) |
| NOx | Ridge, trained once on 2011-13 | 10.35 | 8.34 | 0.04 |
| NOx | Gradient Boosting, walk-forward retraining | **6.69** | 4.69 | **0.60** |
| CO | Ridge, trained once on 2011-13 | 1.39 | 0.83 | 0.595 |
| CO | Gradient Boosting, walk-forward retraining | **1.26** | 0.63 | **0.671** |

RMSE and MAE are in the unit of the target (MWh for TEY, mg/m³ for NOx and CO).

### Key findings

- **Weather and load drive emissions.** NOx falls when the air is hotter (Spearman correlation -0.58). CO is highest at low load (median 3.55 mg/m³ at low load against 1.26 at high load).
- **TEY is almost the same signal as the compressor pressures.** It correlates 0.99 with CDP and 0.96 with GTEP, so the sensor model works as a "soft sensor". It is not a forecast.
- **The weather alone cannot predict TEY**, because the load is set by the operators.
- **NOx drifted.** The yearly average was 67.6, 68.8 and 70.0 mg/m³ in 2011-2013, then 60.1 and 59.9 mg/m³ in 2014-2015. Tuning the model settings did not help (RMSE 12.68 before, 12.86 after). Walk-forward retraining did.
- **The 90% prediction intervals are too narrow.** On the test years they covered only 64.0% of the NOx values and 78.9% of the CO values, so they should not be used as safety limits.

<p align="center">
  <img src="outputs/16_static_vs_walkforward.png" width="85%" alt="Static model against walk-forward retraining">
</p>

---

## Dataset

- **Source:** [Gas Turbine CO and NOx Emission Data on Kaggle](https://www.kaggle.com/datasets/muniryadi/gasturbine-co-and-nox-emission-data). It is a copy of the UCI dataset [Gas Turbine CO and NOx Emission Data Set](https://archive.ics.uci.edu/dataset/551), licensed CC BY 4.0. Please credit the original authors if you reuse it.
- **Size:** 36,733 hourly records from 2011 to 2015, 11 sensor variables.
- **Plant:** a gas turbine power plant in Turkey.

| Variable | Meaning |
|---|---|
| AT | Ambient temperature |
| AP | Ambient pressure |
| AH | Ambient humidity |
| AFDP | Air filter difference pressure |
| GTEP | Gas turbine exhaust pressure |
| TIT | Turbine inlet temperature |
| TAT | Turbine after temperature |
| TEY | Turbine energy yield (MWh) |
| CDP | Compressor discharge pressure |
| CO | Carbon monoxide (mg/m³) |
| NOx | Nitrogen oxides (mg/m³) |

The dataset is **not stored in this repository**. Download it from Kaggle (see "How to run").

---

## What was done

### 1. Data cleaning
- Removed the extra index column.
- Checked for missing values: none.
- Found **7 duplicate rows**, which come in runs of identical readings (frozen sensor values), and removed them. 36,726 rows were left.
- Found **478 humidity readings above 100%** (maximum 100.2), which is physically impossible, and capped them at 100%.
- Checked outliers with the IQR rule and z-scores. They were **kept** because they are real operating events. CO is very skewed, so it was modelled on a log scale.
- The file has **no date column**. The year was rebuilt from the row counts of the original yearly files (7,411 / 7,628 / 7,152 / 7,158 / 7,384). A seasonal temperature check confirmed the row order.
- Added features: temperature drop across the turbine (`dT = TIT - TAT`), log of CO, load bands (low, medium, high from TEY) and temperature bands.

### 2. Exploratory analysis
Distributions, box plots, time series, yearly summaries, correlation heatmaps (Pearson and Spearman), load-band analysis, ambient effects, the air filter pressure loss, and statistical tests (Kruskal-Wallis, Spearman, Mann-Whitney).

<p align="center">
  <img src="outputs/07_key_relationships.png" width="85%" alt="Key relationships between sensors and targets">
</p>

### 3. Machine learning
- **Split by time, not at random:** train on 2011-2013 (22,191 rows) and test on 2014-2015 (14,535 rows).
- **Models:** mean baseline, Ridge regression, Random Forest, Gradient Boosting.
- **Validation:** time-ordered cross-validation with 5 folds on the training years.
- **Four tasks:** TEY from weather only, TEY from all sensors, NOx, and CO (log scale).
- **Metrics:** RMSE, MAE, median absolute error, R² and MAPE.

### 4. Error verification
A good score on one number can hide problems, so the errors were checked in several ways:
- Predicted against actual plots.
- Residual diagnostics: residuals against predictions, histogram, Q-Q plot and Durbin-Watson test.
- Error by year, load band and temperature band.
- Permutation importance, to see which sensors really matter.
- Hyperparameter tuning with time-ordered cross-validation (it did not fix the NOx problem).
- **Walk-forward retraining:** retrain every 500 hours on the latest 7,000 hours. This fixed most of the drift.
- **Prediction intervals** (5% and 95% quantile models) and a check of their real coverage.

<p align="center">
  <img src="outputs/14_feature_importance.png" width="85%" alt="Permutation feature importance">
</p>

---

## Limitations

- The weather-only TEY model has **no real skill**, and the sensor-based TEY model is a soft sensor, not a forecast.
- The dataset does **not explain why NOx dropped in 2014**. A change in operation, combustion tuning or maintenance are possible, but this cannot be shown from the data.
- The year was rebuilt from row counts, not from real timestamps. The seasonal check supports it, but it is an assumption.
- The prediction intervals are too narrow and should not be relied on.
- This is data from one plant. The results may not apply to other turbines.

## Possible next steps

- Add a monitoring rule that flags when the error stays high for several days, and trigger automatic retraining.
- Use conformal prediction to get intervals that really cover 90%.
- Find out what changed at the start of 2014 and add it as a feature.

---

## Repository structure

```
.
├── gas_turbine_emissions.ipynb      # full analysis, cell by cell
├── Gas_Turbine_Emissions_Report.pdf # 6-page written report
├── outputs/                         # charts (PNG), cleaned data and saved models
└── README.md
```

## How to run

1. Download `gt_full.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/muniryadi/gasturbine-co-and-nox-emission-data) and put it next to the notebook. On Kaggle the notebook finds it automatically.
2. Install the libraries:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib
```

3. Open `gas_turbine_emissions.ipynb` and run all cells. The full run takes a few minutes, mostly the cross-validation and the walk-forward retraining.
4. The charts, the cleaned data and the saved models are written to the `outputs/` folder.

The project was developed with Python 3.12, pandas, scikit-learn and seaborn.

## Skills shown

Data cleaning, exploratory data analysis, statistics, time-aware machine learning, model validation, error analysis, drift detection, and clear communication of results.

## Author

**[Your name]**, mechanical engineer and data scientist.
[LinkedIn / email / portfolio link]

## Licence and credits

- Code: add a licence file (for example MIT).
- Data: Gas Turbine CO and NOx Emission Data Set, UCI Machine Learning Repository, CC BY 4.0. Credit the original authors, as listed on the [UCI page](https://archive.ics.uci.edu/dataset/551).
