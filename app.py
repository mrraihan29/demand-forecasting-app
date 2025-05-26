# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go # For interactive plots
import os

# --- App Configuration ---
st.set_page_config(page_title="🧠 Demand Forecasting Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar Info ---
st.sidebar.title("📊 Model & Data Info")
st.sidebar.markdown("Upload custom datasets and view model performance.")

# --- Load Functions ---

@st.cache_data # Cache data loading to prevent re-running on each interaction
def load_data(path):
    """Loads the historical sales data."""
    df = pd.read_csv(path, parse_dates=True, index_col='Date')
    return df

@st.cache_resource # Cache model loading
def load_model(path):
    """Loads the trained Random Forest model."""
    return joblib.load(path)

def create_features(df):
    """Creates temporal and lagged features for the given DataFrame."""
    # Ensure 'Sales' column is present before creating lags
    if 'Sales' not in df.columns:
        raise ValueError("DataFrame must contain a 'Sales' column for feature creation.")

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week # Corrected: Removed .astype(int)
    df['quarter'] = df.index.quarter
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)

    # Lagged features - these will introduce NaNs at the beginning
    df['Sales_Lag_1'] = df['Sales'].shift(1)
    df['Sales_Lag_7'] = df['Sales'].shift(7)
    df['Sales_Lag_365'] = df['Sales'].shift(365)
    
    return df # We will drop NaNs strategically later, not here, for full history


def generate_forecast(model, historical_df, forecast_horizon_days, features_cols):
    """
    Generates future sales forecasts using the trained model.
    Handles lagged features by recursively using predicted values.
    """
    last_historical_date = historical_df.index.max()
    future_dates = pd.date_range(start=last_historical_date + timedelta(days=1),
                                periods=forecast_horizon_days,
                                freq='D')

    forecast_df = pd.DataFrame(index=future_dates)
    forecast_df['Sales'] = np.nan # Placeholder for predicted sales

    # Create a combined DataFrame for seamless lag lookups during recursive forecasting
    # IMPORTANT: Use .copy() to avoid modifying the original historical_df
    # Ensure 'Sales' column is clean for future dates initially
    combined_df_for_lags = pd.concat([historical_df.copy(), pd.DataFrame(index=future_dates, columns=['Sales'])])

    for i, current_date in enumerate(future_dates):
        temp_df_row = pd.DataFrame(index=[current_date])
        temp_df_row['year'] = current_date.year
        temp_df_row['month'] = current_date.month
        temp_df_row['day_of_week'] = current_date.dayofweek
        temp_df_row['day_of_year'] = current_date.dayofyear
        temp_df_row['week_of_year'] = current_date.isocalendar().week # Corrected
        temp_df_row['quarter'] = current_date.quarter
        temp_df_row['is_month_start'] = int(current_date.is_month_start)
        temp_df_row['is_month_end'] = int(current_date.is_month_end)

        # Handle lagged features
        # Sales_Lag_1: Prioritize value from combined_df_for_lags (which will have previous day's forecast)
        lag_1_date = current_date - timedelta(days=1)
        # Use .get() with a fallback or .loc[..., 'Sales'] if .index.contains()
        if lag_1_date in combined_df_for_lags.index:
            temp_df_row['Sales_Lag_1'] = combined_df_for_lags.loc[lag_1_date, 'Sales']
        else:
            # Fallback for very first forecast date if lag_1_date is outside combined_df_for_lags initial range
            temp_df_row['Sales_Lag_1'] = historical_df['Sales'].iloc[-1] if not historical_df.empty else 0


        # Sales_Lag_7: Prioritize value from combined_df_for_lags
        lag_7_date = current_date - timedelta(days=7)
        if lag_7_date in combined_df_for_lags.index:
            temp_df_row['Sales_Lag_7'] = combined_df_for_lags.loc[lag_7_date, 'Sales']
        else:
            temp_df_row['Sales_Lag_7'] = historical_df['Sales'].iloc[-7] if len(historical_df) >= 7 else historical_df['Sales'].mean()


        # Sales_Lag_365: Strictly from historical_df. If outside historical range, use mean.
        lag_365_date = current_date - timedelta(days=365)
        if lag_365_date in historical_df.index:
            temp_df_row['Sales_Lag_365'] = historical_df.loc[lag_365_date, 'Sales']
        else:
            temp_df_row['Sales_Lag_365'] = historical_df['Sales'].mean() # Fallback for very long horizons beyond historical 365-day lag


        # Ensure feature order for prediction
        X_predict = temp_df_row[features_cols]
        
        # Predict sales for current date
        # Catch ValueError if model expects 2D array and gets 1D (e.g., from a single row DataFrame)
        try:
            predicted_sales = model.predict(X_predict)[0]
        except ValueError:
            predicted_sales = model.predict(X_predict.values.reshape(1, -1))[0] # Reshape if necessary

        # Round to nearest integer for sales quantity
        predicted_sales = int(round(predicted_sales))
        
        # Update forecast_df and combined_df_for_lags with the new prediction
        forecast_df.loc[current_date, 'Sales'] = predicted_sales
        combined_df_for_lags.loc[current_date, 'Sales'] = float(predicted_sales)  # Store as float for consistency in lags
                                                                                  # and avoid repeated astype(int) issues


    return forecast_df['Sales']

def calculate_inventory_metrics(forecast_series, lead_time_days, service_level_percent, current_inventory, replenishment_period_days, model_rmse):
    """
    Calculates Reorder Point (ROP) and Order Quantity (Q) based on forecast and parameters.
    """
    # Z-score for service level
    Z_scores = {90: 1.28, 95: 1.645, 99: 2.33}
    Z = Z_scores.get(service_level_percent, 1.645) # Default to 95%

    # 1. Reorder Point (ROP)
    if len(forecast_series) < lead_time_days:
        st.warning(f"Forecast horizon ({len(forecast_series)} days) is less than lead time ({lead_time_days} days). ROP calculation might be inaccurate.")
        # If forecast is too short for lead time, use overall mean for demand during lead time, or average of available
        avg_demand_lead_time = forecast_series.mean() if not forecast_series.empty else 0
    else:
        avg_demand_lead_time = forecast_series.head(lead_time_days).mean()
    
    safety_stock = Z * model_rmse * np.sqrt(lead_time_days)
    safety_stock = max(0, safety_stock) # Ensure non-negative

    rop = (avg_demand_lead_time * lead_time_days) + safety_stock
    rop = int(np.ceil(rop)) if not np.isnan(rop) else 0 # Ensure ROP is integer, handle NaN if avg_demand is NaN

    # 2. Order Quantity (Q)
    if len(forecast_series) < replenishment_period_days:
        st.warning(f"Forecast horizon ({len(forecast_series)} days) is less than replenishment period ({replenishment_period_days} days). Order quantity covers available forecast.")
        forecasted_demand_for_order = forecast_series.sum()
    else:
        forecasted_demand_for_order = forecast_series.head(replenishment_period_days).sum()
    
    order_quantity = int(np.ceil(forecasted_demand_for_order)) if not np.isnan(forecasted_demand_for_order) else 0

    return {
        'reorder_point': rop,
        'order_quantity': order_quantity,
        'avg_demand_lead_time': avg_demand_lead_time,
        'safety_stock': int(np.ceil(safety_stock))
    }

# --- Main Streamlit Application Logic ---

# --- File Uploads (Moved to top level to ensure variables are defined) ---

# Use default paths
default_data_path = "data/raw/synthetic_seasonal_sales.csv"
default_model_path = "models/random_forest_forecasting_model.pkl"

st.sidebar.subheader("📂 Upload Custom Files")

# Assign the uploader widgets' outputs directly to the variables
uploaded_data = st.sidebar.file_uploader("Upload Custom CSV (with Date & Sales)", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Upload Custom Trained Model (.pkl)", type=["pkl"])

# Now, handle the loading logic based on whether files were uploaded
data_to_load = uploaded_data if uploaded_data else default_data_path
model_to_load = uploaded_model if uploaded_model else default_model_path

# Load data
df_original = load_data(data_to_load)

# Load model
model = load_model(model_to_load)

# --- Check if default files exist ---
if not os.path.exists(default_data_path) and not uploaded_data:
    st.error(f"Default data file not found at '{default_data_path}' and no custom file uploaded. Please ensure it is generated by running 01_data_generation_eda.ipynb or upload one.")
    st.stop()
if not os.path.exists(default_model_path) and not uploaded_model:
    st.error(f"Default model file not found at '{default_model_path}' and no custom model uploaded. Please ensure it is trained and saved by running 02_model_training.ipynb or upload one.")
    st.stop()

# --- Check if data and model are loaded ---
if df_original is None or model is None:
    st.info("Waiting for data and model to be loaded or uploaded. Please ensure files are valid.")
    st.stop() # Stop the app if crucial files are missing or invalid

# Apply features to a copy of the original DataFrame
df_processed = create_features(df_original.copy())

# Hardcoded RMSE from 02_model_training.ipynb for safety stock calculation
# (Note: In a production system, this would ideally be part of the saved model's metadata)
model_rmse = 20.83 # From previous notebook's evaluation

# Define features for prediction (must match order and names from training)
features = ['year', 'month', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
            'is_month_start', 'is_month_end', 'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_365']

# --- Tab Navigation ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📦 Inventory", "⚙️ Settings", "📚 Guide & Tutorial"])

# --- Tab 1: Forecast ---
with tab1:
    st.header("📈 Demand Forecasting")
    st.markdown("Forecast future product demand using a trained Random Forest model.")

    st.subheader("Historical Sales Data:")
    st.plotly_chart(go.Figure(data=[go.Scatter(x=df_original.index, y=df_original['Sales'], mode='lines', name='Historical Sales')]).update_layout(height=400, template='plotly_dark'), use_container_width=True)

    forecast_days = st.slider("Select Forecast Horizon (Days)", 7, 180, 30, step=7)
    
    if st.button("🔮 Generate Forecast"):
        # Ensure df_processed is ready for feature extraction for forecasting
        # For forecasting, we need the *full* historical_df (df_original) to derive lags correctly.
        # The create_features in app.py returns df with NaNs, which is not what we want for generate_forecast historical_df.
        # Let's ensure historical_df passed to generate_forecast is the original loaded df *before* NaN dropping.
        with st.spinner(f"Generating {forecast_days}-day forecast..."):
            forecast_series = generate_forecast(model, df_original, forecast_days, features)
            
            st.subheader(f"Forecasted Sales for Next {forecast_days} Days:")
            # --- Plotly Forecast Chart ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_original.index, y=df_original['Sales'], mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines', name='Forecast',
                                    line=dict(color='firebrick', dash='dash')))
            fig.update_layout(title="📊 Historical vs Forecasted Sales", xaxis_title="Date", yaxis_title="Sales",
                              template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)

            # --- Forecast Table & Download ---
            st.subheader("📥 Forecast Table")
            st.dataframe(forecast_series.reset_index().rename(columns={'index': 'Date', 'Sales': 'Forecasted Sales'}))

            csv = forecast_series.reset_index().to_csv(index=False).encode()
            st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")

            st.session_state['forecast_series'] = forecast_series # Store in session state

    # Display last generated forecast if it exists in session state
    elif 'forecast_series' in st.session_state:
        st.subheader("Last Generated Forecast:")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_original.index, y=df_original['Sales'], mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=st.session_state['forecast_series'].index, y=st.session_state['forecast_series'], mode='lines', name='Forecast',
                                line=dict(color='firebrick', dash='dash')))
        fig.update_layout(title="📊 Historical vs Last Forecasted Sales", xaxis_title="Date", yaxis_title="Sales",
                          template='plotly_dark', height=500)
        st.plotly_chart(fig, use_container_width=True)


# --- Tab 2: Inventory Optimizer ---
with tab2:
    st.header("📦 Inventory Optimization")
    st.markdown("This section calculates the optimal Reorder Point (ROP) and recommended Order Quantity based on your forecasted demand and inventory parameters.")

    if 'forecast_series' not in st.session_state:
        st.warning("Please generate a demand forecast first in the '📈 Forecast' tab to enable inventory optimization.")
    else:
        forecast = st.session_state['forecast_series']
        
        st.subheader("Input Inventory Parameters:")
        col1, col2 = st.columns(2)
        with col1:
            inventory = st.number_input("Current Inventory Level (Units)", 0, 5000, 100, help="The current number of units you have in stock.")
            lead_time = st.number_input("Lead Time (days) - Time from order to delivery", 1, 30, 7, help="The number of days it takes for a new order to arrive after it's placed.")
        with col2:
            service_level = st.selectbox("Desired Service Level (%)", [90, 95, 99], index=1, help="The probability of not running out of stock. Higher service level means more safety stock.")
            replenishment_days = st.number_input("Replenishment Period (days) - How many days of demand to cover with each order", 7, 180, 30, help="The duration (in days) for which the new order is intended to cover forecasted demand.")

        # RMSE from training (hardcoded as per our discussion)
        model_rmse = 20.83 
        Z = {90: 1.28, 95: 1.645, 99: 2.33}.get(service_level, 1.645)

        # Calculate inventory metrics
        inventory_results = calculate_inventory_metrics(
            forecast,
            lead_time,
            service_level,
            inventory, # Passing current_inventory
            replenishment_days,
            model_rmse
        )

        st.subheader("Optimization Results:")
        st.metric("🧮 Reorder Point (ROP)", f"{inventory_results['reorder_point']} units")
        st.info(f"This is the inventory level at which you should place a new order. It accounts for demand during lead time and provides a safety buffer. Based on an average daily demand of **{inventory_results['avg_demand_lead_time']:.2f} units** during the {lead_time}-day lead time.")
        
        st.metric("📦 Recommended Order Quantity", f"{inventory_results['order_quantity']} units")
        st.info(f"This is the recommended amount to order. It is calculated to cover the **forecasted demand for the next {replenishment_days} days**.")

        st.markdown("---")
        st.subheader("Detailed Breakdown:")
        with st.expander("Understanding Safety Stock"):
            st.write(f"**Calculated Safety Stock: {inventory_results['safety_stock']} units**")
            st.markdown("""
            Safety Stock is an extra quantity of inventory held to prevent stockouts due to variability in demand or lead time. It acts as a buffer against unforeseen fluctuations.
            
            Our Safety Stock is calculated based on:
            * **Desired Service Level:** Your chosen service level (e.g., 95%) determines how confident you want to be in avoiding a stockout. A higher service level requires more safety stock.
            * **Forecast Error (RMSE):** The Root Mean Squared Error (RMSE) of our demand forecasting model (which is {model_rmse:.2f} for our model) serves as a proxy for the variability or uncertainty in demand.
            * **Lead Time:** More variability accumulates over longer lead times, so safety stock increases with lead time.
            
            **Formula:** $ \\text{Safety Stock} = Z \\times \\text{RMSE} \\times \\sqrt{\\text{Lead Time}} $
            """)
            
        with st.expander("When to Order? (Current Inventory vs. ROP)"):
            if inventory <= inventory_results['reorder_point']:
                st.error(f"⚠️ **ACTION REQUIRED!** Your Current Inventory ({inventory} units) is at or below the calculated Reorder Point ({inventory_results['reorder_point']} units). You should consider placing an order of **{inventory_results['order_quantity']} units** soon to prevent stockouts.")
            else:
                st.success(f"✅ Your Current Inventory ({inventory} units) is above the calculated Reorder Point ({inventory_results['reorder_point']} units). No immediate order is needed, but continue monitoring.")
            st.markdown("""
            The relationship between your current inventory and the Reorder Point is crucial:
            * **Current Inventory $\\le$ Reorder Point:** It's time to place an order.
            * **Current Inventory $>$ Reorder Point:** You still have enough stock; continue monitoring.
            """)
        
        st.caption("Inventory calculations are based on statistical methods and forecasted demand. Actual results may vary.")

# --- Tab 3: Settings / Metrics ---
with tab3:
    st.header("⚙️ Model Performance Metrics")
    st.markdown("Performance metrics from the training phase (02_model_training.ipynb):")

    st.metric("📏 RMSE", "20.83")
    st.metric("📉 MAE", "17.07") # Corrected value
    st.metric("📈 R² Score", "0.74") # Corrected value

    st.subheader("📌 Notes")
    st.markdown("""
    - **Lag Features:** The model utilizes **Sales_Lag_1**, **Sales_Lag_7**, and **Sales_Lag_365** (sales from 1 day, 7 days, and 365 days ago, respectively) to capture temporal dependencies.
    - **Recursive Forecasting:** Future forecasts are generated recursively, meaning each day's prediction is used as a lagged input for subsequent day's predictions within the forecast horizon.
    - **Model Compatibility:** When uploading custom models, ensure they are trained with the exact same feature set and order as defined in this application.
    """)
  
# --- Tab 4: Guide & Tutorial ---
with tab4:
    st.header("📚 User Guide & Scientific Foundations")
    st.caption("This guide aims to provide transparency and educational value. For production-grade systems, further customization and validation would be required.")
    st.markdown("""
Welcome to the **Demand Forecasting & Inventory Optimization Dashboard** — a practical and educational tool that combines machine learning and supply chain logic to help you forecast demand and make smarter inventory decisions.

This guide walks you through both *how to use the app* and the *principles behind its design*.
""")

    st.subheader("1. 🛠️ Using the App")

    st.markdown("""
This dashboard is organized into four main tabs:

| Tab | Description |
|-----|-------------|
| **📈 Forecast** | Visualize past sales and generate machine learning-based demand forecasts |
| **📦 Inventory** | Calculate reorder points and order quantities using forecast results |
| **⚙️ Settings** | Review model performance metrics |
| **📚 Guide** | Learn how the system works and the science behind it |

**Sidebar Functions:**
- **Upload CSV:** Bring your own sales history (`Date` + `Sales`) and model (`.pkl`).
- **Fallback Defaults:** If nothing is uploaded, synthetic data and a pre-trained model are used.
""")

    st.subheader("2. 🤖 How Forecasting Works")

    st.markdown("""
This app uses a **Random Forest Regressor**, trained on historical daily sales. Forecasting is powered by robust **feature engineering** and **recursive prediction**.

### 🔍 Feature Engineering
Since models don’t understand raw dates, we convert them into informative numeric features:

- **Temporal Features**:  
    `year`, `month`, `day_of_week`, `day_of_year`, `week_of_year`, `quarter`, `is_month_start`, `is_month_end`

- **Lag Features**:  
    Sales from previous days like `Sales_Lag_1`, `Sales_Lag_7`, `Sales_Lag_365`  
    *(these provide context about past demand patterns)*

### 🔄 Recursive Forecasting
To predict 30 days ahead:
1. Day 1 forecast uses actual past data.
2. Day 2 forecast uses Day 1's prediction as part of its input.
3. The chain continues recursively.

This lets the model simulate a realistic forecast horizon.

""")

    st.plotly_chart(go.Figure(data=[
        go.Scatter(x=df_original.index, y=df_original['Sales'], mode='lines', name='Historical Sales')
    ]).update_layout(title="📊 Historical Sales Pattern", template='plotly_dark', height=300), use_container_width=True)

    st.caption("The model learns seasonality and long-term trends from historical sales.")

    st.subheader("3. 📦 Inventory Optimization Logic")

    st.markdown("""
Once a forecast is generated, the app calculates:

- 📍 **Reorder Point (ROP)** — When to place an order  
- 📦 **Order Quantity (Q)** — How much to order

These are based on your business settings (inventory, lead time, service level).

### 📐 Formulas

**Safety Stock (SS):**
$$
SS = Z \\times RMSE \\times \\sqrt{Lead\\_Time}
$$

**Reorder Point (ROP):**
$$
ROP = (\\bar{D}_{lead\\_time} \\times Lead\\_Time) + SS
$$

**Order Quantity (Q):**
$$
Q = \\sum_{t=1}^{T} Forecast_t \\quad \\text{(for the replenishment period)}
$$

**Where:**
- $Z$: Z-score for your service level (e.g., 1.645 for 95%)
- $RMSE$: Model forecast error
- $\\bar{D}_{lead\\_time}$: Avg. daily forecast during lead time
""")

    st.markdown("""
These calculations balance **demand risk** with **inventory costs**.

**Inputs You Control:**
- Current Inventory
- Lead Time (days)
- Desired Service Level (e.g., 95%)
- Replenishment Horizon (how many days to cover per order)
""")

    st.subheader("4. 📊 Model Performance Metrics")

    st.markdown("""
We assess model quality using standard regression metrics:

| Metric | Meaning |
|--------|---------|
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **MAE** | Mean Absolute Error — intuitive, less sensitive to outliers |
| **R² Score** | Explains how much variance is captured by the model (closer to 1 is better) |

These help you decide whether the forecast is reliable enough for operational use.
""")

    st.subheader("5. ⚠️ Limitations & Future Enhancements")

    st.markdown("""
This app provides a simplified yet effective forecasting engine. However, it's not a complete enterprise solution.

**Current Limitations:**
- Works on a **single product** only
- Forecast assumes **normally distributed error**
- Lead time & parameters are **fixed/static**
- Doesn't incorporate **cost optimization models**
- Ignores **external drivers** like holidays, price changes, or promotions

**Future Ideas:**
- Multi-SKU forecasting support
- Include **holidays & promotions** as features
- Implement **(s, S)** policies and EOQ models
- Integrate with **live databases** or **APIs**
- Add **prediction intervals** for uncertainty quantification
""")

st.markdown("---")
st.caption("🧠 Built for transparency, education, and practical insight. For real-world deployment, enhancements and validations are recommended.")
st.caption("Developed by Muhammad Rizky Raihan | Powered by Streamlit + Scikit-Learn")