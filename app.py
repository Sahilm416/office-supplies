import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_excel('store.xls', engine='xlrd')
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        st.stop()
    
    data = data[['Order Date', 'Category', 'Sales']]
    office_supplies_df = data.loc[data['Category'] == 'Office Supplies']
    office_supplies_df = office_supplies_df[['Order Date', 'Sales']]
    office_supplies_df = office_supplies_df.sort_values('Order Date')
    office_supplies_df = office_supplies_df.groupby('Order Date')['Sales'].sum().reset_index()
    office_supplies_df.set_index('Order Date', inplace=True)
    office_supplies_df = office_supplies_df['Sales'].resample('MS').mean()
    return office_supplies_df

# Train SARIMA model
@st.cache_resource
def train_sarima_model(data):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    return results

# Make predictions
def make_predictions(model, start_date, end_date):
    predictions = model.get_forecast(steps=len(pd.date_range(start_date, end_date, freq='MS')))
    return predictions.predicted_mean, predictions.conf_int()

# Streamlit app
st.title('Office Supplies Sales Time Series Forecasting')

# Load data
office_supplies_df = load_data()

# Sidebar
st.sidebar.header('Options')
last_date = office_supplies_df.index[-1]
train_end_date = st.sidebar.date_input('Training End Date', value=last_date - pd.DateOffset(months=12), max_value=last_date)
forecast_months = st.sidebar.slider('Forecast Months', 1, 24, 12)

# Split data
train_df = office_supplies_df.loc[:train_end_date]
test_df = office_supplies_df.loc[train_end_date + pd.DateOffset(days=1):]

# Train model
model = train_sarima_model(train_df)

# Make predictions
forecast_start = train_end_date + pd.DateOffset(months=1)
forecast_end = forecast_start + pd.DateOffset(months=forecast_months-1)
predictions, conf_int = make_predictions(model, forecast_start, forecast_end)

# Plotting
st.subheader('Sales Over Time')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(office_supplies_df.index, office_supplies_df, label='Actual Sales')
ax.plot(predictions.index, predictions, label='Forecast', color='red')
ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
ax.axvline(train_end_date, color='green', linestyle='--', label='Train/Test Split')
ax.legend()
plt.title('Office Supplies Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
st.pyplot(fig)

# Display metrics
st.subheader('Model Performance')
# Align predictions with test data
aligned_predictions = predictions.reindex(test_df.index)
aligned_predictions = aligned_predictions.dropna()
test_df_aligned = test_df.loc[aligned_predictions.index]

if not aligned_predictions.empty:
    mse = np.mean((aligned_predictions - test_df_aligned)**2)
    mae = np.mean(np.abs(aligned_predictions - test_df_aligned))
    rmse = np.sqrt(mse)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
else:
    st.warning("Not enough test data to calculate performance metrics.")

# Decomposition plot
st.subheader('Time Series Decomposition')
decomposition = seasonal_decompose(office_supplies_df, model='additive', period=12)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Observed')
decomposition.trend.plot(ax=ax2)
ax2.set_title('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
decomposition.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
st.pyplot(fig)

# Forecast table
st.subheader('Forecast')
forecast_df = pd.DataFrame({
    'Date': predictions.index,
    'Predicted Sales': predictions.values,
    'Lower CI': conf_int.iloc[:, 0].values,
    'Upper CI': conf_int.iloc[:, 1].values
})
st.dataframe(forecast_df)

# Download forecast as CSV
csv = forecast_df.to_csv(index=False)
st.download_button(
    label="Download Forecast as CSV",
    data=csv,
    file_name="office_supplies_sales_forecast.csv",
    mime="text/csv"
)