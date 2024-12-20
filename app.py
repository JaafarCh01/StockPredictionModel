import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model
model = load_model("AAPL_model.h5")

# Helper function to prepare data
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_test = []
    for i in range(time_step, len(scaled_data)):
        x_test.append(scaled_data[i-time_step:i, 0])
    
    return np.array(x_test), scaler

# Streamlit App
st.title("AAPL Stock Price Prediction")
st.write("This application predicts AAPL stock prices using a pre-trained LSTM model.")

# Sidebar
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))

# Fetch data
if st.sidebar.button("Fetch Data"):
    st.write(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if not df.empty:
        st.write("### Historical Stock Prices")
        st.line_chart(df['Close'])
        
        # Prepare data for prediction
        data = df['Close']['AAPL'].values.reshape(-1, 1)
        x_test, scaler = prepare_data(data)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
        # Predict
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        
        # Visualize
        st.write("### Predicted Stock Prices")
        df['Predicted'] = np.nan
        df.loc[df.index[60:], 'Predicted'] = predictions.flatten()
        
        # Reset column index for plotting
        df.columns = df.columns.get_level_values(0)
        
        # Debug: Check the DataFrame columns and first few rows
        st.write("DataFrame Columns:", df.columns)
        st.write("DataFrame Head:", df.head())
        
        # Plot the data
        st.line_chart(df[['Close', 'Predicted']].dropna())
    else:
        st.error(f"No data found for ticker {ticker}. Please try again.")
