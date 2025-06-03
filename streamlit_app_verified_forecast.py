
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from fredapi import Fred
from datetime import datetime

# --- Load Data ---
df = pd.read_csv("starbucks_financials_expanded.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# --- Get Live CPI from FRED ---
fred = Fred(api_key="841ad742e3a4e4bb2f3fcb90a3d078fb")
cpi_series = fred.get_series('CPIAUCSL')
current_cpi = cpi_series.iloc[-1]
prev_cpi = cpi_series.iloc[-13]
cpi_percent = ((current_cpi - prev_cpi) / prev_cpi) * 100

st.title("📊 Starbucks Revenue Forecasting App (Backtest)")
st.write(f"### Current CPI (Year-over-Year): {cpi_percent:.2f}%")

# --- User Inputs ---
st.sidebar.header("User Inputs")
cpi_input = st.sidebar.slider("Adjusted CPI (%)", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
txn_input = st.sidebar.slider("Expected Transactions", 800, 1200, 1000, step=10)
mkt_input = st.sidebar.slider("Expected Marketing Spend ($M)", 300, 800, 500, step=10)

# --- Forecast Period ---
train_data = df.loc[:'2022-12-31']
test_data = df.loc['2023-01-01':]

# --- Train ARIMAX Model 3: CPI + Transactions + Marketing Spend ---
endog_train = train_data['revenue']
exog_train = train_data[['cpi', 'transactions', 'marketing_spend']]

model = SARIMAX(endog_train, exog=exog_train, order=(1, 1, 1))
results = model.fit(disp=False)

# --- Forecast 2023 Revenue using actual data ---
exog_forecast = test_data[['cpi', 'transactions', 'marketing_spend']]
forecast = results.get_prediction(start=test_data.index[0], end=test_data.index[-1], exog=exog_forecast)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# --- Plot Actual vs Forecasted Revenue for Full Timeline ---
st.subheader("📈 Revenue Forecast (Backtest with Model 3)")
fig1, ax1 = plt.subplots()
df['revenue'].plot(ax=ax1, label='Actual Revenue', color='blue')
forecast_mean.plot(ax=ax1, label='Forecasted Revenue (2023)', color='green')
ax1.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='green', alpha=0.3)
ax1.set_ylabel("Revenue")
ax1.legend()
st.pyplot(fig1)

# --- Forecast Error Analysis ---
actual = test_data['revenue']
errors = actual - forecast_mean
mae = np.mean(np.abs(errors))
mape = np.mean(np.abs(errors / actual)) * 100

st.subheader("📊 Forecast Error Analysis")
st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")

# --- Regression Model: Expected Revenue from Project 2 Variables ---
X_reg = df[['cpi', 'transactions', 'marketing_spend']]
y_reg = df['revenue']
reg_model = LinearRegression().fit(X_reg, y_reg)
df['expected_revenue'] = reg_model.predict(X_reg)

# --- Plot: Expected Revenue vs Actual Revenue and Expenses ---
st.subheader("📊 Expected Revenue vs Actual Revenue and Expenses")
fig2, ax2 = plt.subplots()
df['revenue'].plot(ax=ax2, label='Actual Revenue', color='blue')
df['expected_revenue'].plot(ax=ax2, label='Expected Revenue (Regression)', linestyle='--', color='green')
df['expenses'].plot(ax=ax2, label='Actual Expenses', color='red')
ax2.set_ylabel("USD ($)")
ax2.legend()
st.pyplot(fig2)

# --- Quick Graph Summary ---
st.markdown("""### 📌 Graph Summary
The second chart shows actual revenue, expected revenue based on CPI, transactions, and marketing spend, and actual expenses. Large or persistent gaps between actual and expected revenue may signal risk areas worth further audit investigation.
""")

# --- Enhanced AI Summary for Audit Committee ---
st.subheader("🧠 AI Summary for Audit Committee")
summary = f"""
**Summary**: This app uses ARIMAX Model 3 (CPI, transactions, and marketing spend) to forecast 2023 revenue. The model achieved a MAPE of {mape:.2f}%, suggesting reasonable accuracy. A regression-based benchmark also shows the expected revenue based on economic drivers. Discrepancies between expected and actual revenue, particularly during periods of high CPI or aggressive marketing spend, may indicate a **risk of revenue overstatement**. These tools help the audit committee evaluate whether Starbucks' reported revenue aligns with underlying drivers or may be overstated due to overly aggressive inputs.
"""
st.markdown(summary)
