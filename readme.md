# Forest Fire Weather Index (FWI) Prediction

A machine learning project that analyzes the Algerian Forest Fires dataset, applies data cleaning and feature engineering, compares regression models, and provides a Flask web application for predicting the **Fire Weather Index (FWI)** from environmental and fire-weather measurements.

## Project Overview

Forest fire behavior is influenced by weather conditions and fuel-moisture indicators. This project explores how these variables relate to the Fire Weather Index (FWI), a numerical fire-weather indicator.

The project includes:
- Exploratory data analysis of the Algerian Forest Fires dataset.
- Data cleaning and preparation of environmental measurements.
- Feature engineering, including categorical encoding, removal of date columns, and elimination of highly correlated features.
- Standardization of model inputs with `StandardScaler`.
- Comparison of multiple regression algorithms.
- A Flask application that accepts environmental inputs and displays an FWI prediction.

**Task type:** Regression. The application predicts a numerical FWI value.

## Technology Stack

- **Language:** Python
- **Data manipulation:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn
- **Web framework:** Flask
- **Frontend:** HTML and Jinja2 templates
- **Model persistence:** Pickle
- **Development and analysis:** Jupyter Notebook


## Dataset

The notebook uses the **Algerian Forest Fires dataset** and creates a cleaned file named `Algerian_forest_fires_cleaned_dataset.csv`.

The cleaned CSV contains **243 rows and 15 columns**:

- Date fields: `day`, `month`, `year`
- Environmental and fire-weather features: `Temperature`, `RH`, `Ws`, `Rain`, `FFMC`, `DMC`, `DC`, `ISI`, `BUI`
- Target: `FWI`
- Additional fields: `Classes`, `Region`


## Data Cleaning and Feature Engineering

The notebook prepares the dataset before training:

1. **Inspect and clean the data:** Examine the dataset structure and missing values, and remove incomplete records when preparing the cleaned dataset.
2. **Prepare the target:** Use `FWI` as the regression target.
3. **Encode categorical data:** Convert the `Classes` feature into a numeric representation (`not fire` → `0`; the other class value → `1`).
4. **Add region information:** Retain a numeric `Region` indicator to distinguish the two regions represented in the dataset.
5. **Remove date columns from the model inputs:** Drop `day`, `month`, and `year` from the feature matrix.
6. **Separate features and target:** Keep `FWI` as `y` and the remaining candidate predictors as `X`.
7. **Remove highly correlated features:** Identify correlated columns in the training data and drop the selected `corr_features` from both `x_train` and `x_test`. This reduces redundant input information and leaves the final nine model features.
8. **Scale the features:** Fit `StandardScaler` on `x_train` only, then apply the same fitted scaler to `x_test`. This avoids fitting preprocessing parameters on the test set.



The model and the Flask form must use the same nine features in exactly the same order. The saved scaler must be the one fitted on those same features.


## Exploratory Data Analysis

The notebook includes:
- Dataset structure and missing-value inspection.
- Summary statistics.
- Histograms and class-distribution plots.
- Correlation analysis and a heatmap.
- Boxplot analysis of FWI.
- Fire-class counts by month and region.

## Model Training

- **Target (`y`):** `FWI`
- **Final input count:** 9 features after correlated-feature removal.
- **Train/test split:** `train_test_split` with `test_size=0.25` and `random_state=42`.
- **Scaling:** `StandardScaler`, fitted on the training features and reused for test data.
- **Selected inference model:** Ridge Regression, as referenced by the Flask application.

### Models Compared

The notebook compares several regression methods, including Linear Regression, Lasso, Ridge, Elastic Net, RidgeCV, and ElasticNetCV.


## Flask Web Application

The Flask application entry point is `application.py`.

### Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Displays the input form in `templates/index.html`. |
| `/predictdata` | GET | Displays the prediction page without a submitted prediction. |
| `/predictdata` | POST | Reads the submitted measurements, applies the saved scaler, predicts FWI with the saved model, and renders the result in `templates/home.html`. |

The form accepts nine inputs:

`Temperature`, `RH`, `Ws`, `Rain`, `FFMC`, `DMC`, `ISI`, `Classes`, `Region`

The project’s intended inference workflow uses these nine features in the same order as the training pipeline after correlated-feature removal. Keep the form order, preprocessing order, scaler, and model synchronized whenever the feature pipeline changes.
## Author

**Krithik Sagala**


