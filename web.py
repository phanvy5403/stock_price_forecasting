import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
scaler = MinMaxScaler()

# Streamlit Title
st.title("Stock Price Predictor App")

# Input for Stock ID
stock = st.text_input("Enter the Stock ID", "^NSEI")

# Set start and end date for data
from datetime import datetime, timedelta
end = datetime.now()
start = datetime(end.year-10, end.month, end.day)

# Download stock data
data = yf.download(stock, start, end)

# Remove weekends (Saturday and Sunday)
data = data[data.index.dayofweek < 5]

# Load pre-trained model
model = load_model("model.h5")

# Add 'Range' column to data
data['Range'] = data['High'] - data['Low']
data.drop(['Adj Close'], axis=1, inplace=True)

# Display the stock data
st.subheader("Hisotry Stock Data over 10 years")
st.write(data)

# Split data for training and testing
splitting_len = int(len(data) * 0.7)
data_splited = pd.DataFrame(data[splitting_len:])

# Scale the data
data_scaled = scaler.fit_transform(data_splited[['Open', 'High', 'Low', 'Close', 'Volume', 'Range']])

# Prepare dataset for model
dataset = np.array(data_scaled)
X, Y = [], []
for i in range(len(dataset) - 10 - 5 + 1):
    X.append(dataset[i:(i + 10), 1:6])
    Y.append(dataset[(i + 10):(i + 10 + 5), 0])

X, Y = np.array(X), np.array(Y)

# Predictions
predictions = model.predict(X)

# Rescale the data back to original scale
Y = Y * (data['Open'].max() - data['Open'].min()) + data['Open'].min()
predictions = predictions * (data['Open'].max() - data['Open'].min()) + data['Open'].min()

# Prepare data for plotting
plotting_data = pd.DataFrame(
    {
        'original_test_data': Y.reshape(-1),
        'predictions': predictions.reshape(-1)
    }
)

# Plot the original vs predicted prices
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(plotting_data.original_test_data)
plt.plot(plotting_data.predictions)
plt.legend(['Original Open Price', 'Predicted Open Price'])
st.pyplot(fig)

# Prepare the latest 10 days data for prediction of next 5 days
latest_data = data[-10:]
latest_scaled = scaler.transform(latest_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Range']])
latest_scaled = latest_scaled[:, 1:6].reshape(1, 10, 5)

# Predict the next 5 days
next_5_days_prediction = model.predict(latest_scaled)
next_5_days_prediction = next_5_days_prediction * (data['Open'].max() - data['Open'].min()) + data['Open'].min()

# Generate next 5 business days dates
next_5_days_dates = []
current_date = end
while len(next_5_days_dates) < 5:
    current_date += timedelta(days=1)
    if current_date.weekday() < 5:  # Monday to Friday are valid days
        next_5_days_dates.append(current_date)
        
st.subheader("Latest 10 Days Data (Used for Forecasting)")
latest_data_display = latest_data.copy()
latest_data_display.index = latest_data_display.index.strftime('%Y-%m-%d')
st.write(latest_data_display)

# Create a new DataFrame for next 5 days prediction
next_5_days_df = pd.DataFrame(
    {
        'Date': next_5_days_dates,
        'Predicted_Open_Price': next_5_days_prediction.flatten()
    }
)
next_5_days_df['Date'] = next_5_days_df['Date'].dt.strftime('%Y-%m-%d')

# Plot the next 5 days prediction
st.subheader('Predicted Open Prices for Next 5 Business Days')
st.write(next_5_days_df)

# Display the latest 10 days data
