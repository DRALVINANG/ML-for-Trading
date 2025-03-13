# --------------------------------------------------------------------------------------
# Step 1: Install and Import Libraries
# --------------------------------------------------------------------------------------

import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# Step 2: Load and Prepare Data
# --------------------------------------------------------------------------------------

# Load the dataset
df = pd.read_csv('https://gist.githubusercontent.com/DRALVINANG/5821855d6bcce977fc7f7638bb7ea9a3/raw/9d5bf33a581bf81a8319baf9d677eef309a2d7e9/TSLA%2520Stock%2520Price%2520(2020).csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# --------------------------------------------------------------------------------------
# Step 3: Apply Technical Indicators
# --------------------------------------------------------------------------------------

# 3.1 Simple Moving Average (SMA)
df['SMA_5'] = ta.sma(close=df.Close, length=5)

# 3.2 Exponential Moving Average (EMA)
df['EMA_5'] = ta.ema(close=df['Close'], length=5)

# 3.3 Relative Strength Index (RSI)
df['RSI_5'] = ta.rsi(close=df.Close, length=5)

# 3.4 Bollinger Bands
bb = ta.bbands(df.Close, length=14, std=2)
bb.drop(['BBB_14_2.0', 'BBP_14_2.0'], axis=1, inplace=True)
bb.columns = ['Lower', 'Mid', 'Upper']
df = df.join(bb)

# 3.5 Average Directional Index (ADX)
adx = ta.adx(df.High, df.Low, df.Close, length=14)
df = df.join(adx)
df.drop(['DMP_14', 'DMN_14'], axis=1, inplace=True)

# 3.6 Moving Average Convergence Divergence (MACD)
macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df['MACD'] = macd['MACD_12_26_9']

# --------------------------------------------------------------------------------------
# Step 4: Visualize Data with Plots
# --------------------------------------------------------------------------------------

# 4.1 Plot Close Price and SMA_5
plt.style.use('fivethirtyeight')
df[['Close', 'SMA_5']].plot(figsize=(12, 12))
plt.title("Tesla Stock Price and 5-Period SMA (2020)", fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.show()

# 4.2 Plot Close Price, SMA_5, and EMA_5
df[['Close', 'SMA_5', 'EMA_5']].plot(figsize=(12, 12))
plt.title("Tesla Stock Price, 5-Period SMA, and 5-Period EMA (2020)", fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.show()

# 4.3 Plot RSI
df['RSI_5'].plot(figsize=(12, 12))
plt.title("Tesla Stock Price and 5-Period RSI (2020)", fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('RSI', fontsize=12)
plt.show()

# 4.4 Plot Bollinger Bands
df[['Close', 'Lower', 'Mid', 'Upper']].plot(figsize=(12, 6))
plt.title("Tesla Stock Price with Bollinger Bands (2020)", fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.show()

# 4.5 Plot ADX
df['ADX_14'].plot(figsize=(12, 12))
plt.title("Tesla Stock Price and 14-Period ADX (2020)", fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('ADX', fontsize=12)
plt.show()

# 4.6 Plot MACD, Signal Line, and Histogram
plt.figure(figsize=(12, 6))
plt.plot(macd.index, macd['MACD_12_26_9'], label='MACD', color='blue', linewidth=2)
plt.plot(macd.index, macd['MACDs_12_26_9'], label='Signal Line', color='red', linewidth=2)
plt.bar(macd.index, macd['MACDh_12_26_9'], label='Histogram', color='gray', alpha=0.5)
plt.title("Tesla Stock Price - MACD, Signal Line, and Histogram (2020)", fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=90)
plt.legend()
plt.show()

# --------------------------------------------------------------------------------------
# Step 5: Plot Normal Candlestick Chart
# --------------------------------------------------------------------------------------

fig = go.Figure(data=[go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close)])
fig.update_layout(
    title="Tesla Stock Price (2020)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
)

fig.show()

# --------------------------------------------------------------------------------------
# Step 6: Plot Heikin Ashi Candlestick Chart
# --------------------------------------------------------------------------------------

# Compute Heikin Ashi Data
ha_df = ta.ha(open_=df.Open, high=df.High, low=df.Low, close=df.Close)

# Plot Heikin Ashi Candlesticks
fig = go.Figure(data=[go.Candlestick(x=df.index, open=ha_df.HA_open, high=ha_df.HA_high, low=ha_df.HA_low, close=ha_df.HA_close)])
fig.update_layout(
    title="Tesla Stock Price (2020)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
)
fig.show()

# --------------------------------------------------------------------------------------
# Step 7: End of Analysis
# --------------------------------------------------------------------------------------

