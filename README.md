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

## 1. Run locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Re)train the model — only needed if df.pkl / pipe.pkl aren't present,
#    or if you change train_model.py
python train_model.py

# 4. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 2. Deploy for free — Streamlit Community Cloud (recommended, easiest)

1. Push this folder to a **public GitHub repository** (must include
   `app.py`, `requirements.txt`, `df.pkl`, `pipe.pkl` — you don't need to
   push `laptop_data.csv` or `train_model.py`, but it's fine to include them).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set the main file path to
   `app.py`.
4. Click **Deploy**. In a minute or two you'll get a public URL like
   `https://your-app-name.streamlit.app`.

That's it — no server setup, free hosting, auto-redeploys on every git push.

## 3. Alternative: Deploy on Render

1. Push the folder to GitHub (same as above).
2. Go to **https://render.com** → **New** → **Web Service** → connect your repo.
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **Create Web Service**. Render will build and give you a public URL.

## 4. Alternative: Deploy with Docker (any cloud)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & run:
```bash
docker build -t laptop-price-app .
docker run -p 8501:8501 laptop-price-app
```

Push the image to any container host (Render, Railway, Fly.io, AWS/GCP/Azure)
that supports Docker deployments.

## Retraining on new data

Replace `laptop_data.csv` with an updated dataset (same column format) and
re-run:
```bash
python train_model.py
```
This regenerates `df.pkl` and `pipe.pkl`, which `app.py` picks up automatically.
