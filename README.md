# Laptop Price Predictor

A simple Streamlit web app that predicts a laptop's price from its specs
(brand, RAM, CPU, GPU, storage, screen, etc.), using a scikit-learn model
trained on `laptop_data.csv`.

## Files

| File              | Purpose                                                            |
|-------------------|---------------------------------------------------------------------|
| `laptop_data.csv` | Raw dataset                                                        |
| `train_model.py`  | Cleans the data, engineers features, trains the model, saves `.pkl`|
| `df.pkl`          | Cleaned dataframe (used by the app to build dropdown options)      |
| `pipe.pkl`        | Trained scikit-learn Pipeline (preprocessing + RandomForest model) |
| `app.py`          | Streamlit web app                                                  |
| `requirements.txt`| Python dependencies                                                |

The model achieves **R² ≈ 0.89** and **MAE ≈ 0.16** (on log-price) on the test split.

> Note: the original notebook's best model was a `StackingRegressor` that
> included XGBoost. `train_model.py` uses `RandomForestRegressor` instead,
> since it needs no extra dependency beyond scikit-learn and performs
> almost as well. If you have `xgboost` installed, swap it back in —
> `app.py` doesn't need any changes since it just loads `pipe.pkl`.

