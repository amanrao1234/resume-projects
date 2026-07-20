import numpy as np
import pandas as pd
import pickle
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

st.title("💻 Laptop Price Predictor")
st.write(
    "Configure the laptop specs below and get an estimated market price, "
    "based on a machine learning model trained on real laptop listings."
)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    df = pickle.load(open('df.pkl', 'rb'))
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    return df, pipe


df, pipe = load_artifacts()

# ---------------------------------------------------------------------------
# Input widgets
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", sorted(df['Company'].unique()))
    type_name = st.selectbox("Type", sorted(df['TypeName'].unique()))
    ram = st.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    cpu_brand = st.selectbox("CPU", sorted(df['Cpu brand'].unique()))

with col2:
    gpu_brand = st.selectbox("GPU", sorted(df['Gpu brand'].unique()))
    os_name = st.selectbox("Operating System", sorted(df['os'].unique()))
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    ips = st.selectbox("IPS Display", ["No", "Yes"])
    screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)

st.markdown("**Screen Resolution**")
resolution = st.selectbox(
    "Resolution",
    ["1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800",
     "2880x1800", "2560x1600", "2560x1440", "2304x1440"],
    label_visibility="collapsed",
)

st.markdown("**Storage**")
scol1, scol2 = st.columns(2)
with scol1:
    hdd = st.selectbox("HDD (GB)", sorted(df['HDD'].unique()))
with scol2:
    ssd = st.selectbox("SSD (GB)", sorted(df['SSD'].unique()))

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict Price", type="primary", use_container_width=True):
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    x_res, y_res = map(int, resolution.split('x'))
    ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size

    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': ppi,
        'Cpu brand': cpu_brand,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu_brand,
        'os': os_name,
    }])

    predicted_log_price = pipe.predict(query)[0]
    predicted_price = int(np.exp(predicted_log_price))

    st.success(f"### Estimated Price: ₹{predicted_price:,}")
    st.caption(
        "This is a model-generated estimate based on historical listing prices "
        "and may not reflect current market rates."
    )

st.divider()
st.caption("Built with Streamlit · Model: RandomForestRegressor on a scikit-learn Pipeline")
