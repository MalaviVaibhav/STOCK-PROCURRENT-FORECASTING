import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np

# Load your trained pipeline (pickle version)
with open("stock_forecast_pipeline.pkl", "rb") as f:
    pipe = pkl.load(f)

# Load your cleaned dataset for dropdowns
df = pd.read_csv("data.csv")

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Procurement Forecast", layout="centered")
st.title("📦 Stock Procurement Forecasting App")

st.sidebar.markdown("## 🔍 Select Stock Details")

with st.sidebar.expander("Choose parameters", expanded=True):
    # Month dropdown (unique sorted months)
    months = sorted(df['month'].unique())
    month = st.selectbox("Select Month", months)

    # Year dropdown (unique sorted years)
    years = sorted(df['year'].unique())
    year = st.selectbox("Select Year", years)

    # Brand dropdown (unique brands available for selected month and year)
    brands = sorted(df[(df['month'] == month) & (df['year'] == year)]['brandname'].unique())
    brand = st.selectbox("Select Brand", brands)

    # Product dropdown (unique products available for selected brand)
    products = sorted(df[(df['brandname'] == brand)]['productname'].unique())
    product = st.selectbox("Select Product", products)

    # Name dropdown (technical specs for selected product)
    names = sorted(df[(df['productname'] == product)]['name'].unique())
    name = st.selectbox("Select Specification", names)

# --- Prediction Button ---
st.markdown("---")
if st.button("Predict Stock to Procure 📈"):
    # Prepare input dataframe exactly as model expects
    input_data = pd.DataFrame([[month, year, brand, product, name]], 
                              columns=['month', 'year', 'brandname', 'productname', 'name'])
    
    st.markdown("### Your Input")
    st.dataframe(input_data)

    # Predict weight (stock quantity)
    pred_weight = pipe.predict(input_data)
    pred_weight_rounded = int(np.round(pred_weight[0], 2))

    st.markdown("## Suggested Stock to Procure")
    st.success(f"🛒 You should procure approximately **{pred_weight_rounded} units** ")

