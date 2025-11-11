# --------------------------------------------------------------------------------------
# Step 1: Install and Import Libraries
# --------------------------------------------------------------------------------------

import pandas as pd
import pandas_ta_classic as ta  # Using pandas-ta-classic
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

# Preview of pandas-ta functionality
print(help(df.ta))  # Display help documentation for pandas-ta
print('---------------------------------------------------')
print(df.ta.indicators())  # List available indicators in pandas-ta
print('----------------------------------------------------')
print(help(ta.rsi))  # Display help documentation for RSI indicator

# --------------------------------------------------------------------------------------
# Step 3: Apply Technical Indicators
# --------------------------------------------------------------------------------------

# 3.1 Simple Moving Average (SMA)
df['SMA_5'] = ta.sma(df['Close'], length=5)

# 3.2 Exponential Moving Average (EMA)
df['EMA_5'] = ta.ema(df['Close'], length=5)

# 3.3 Relative Strength Index (RSI)
df['RSI_5'] = ta.rsi(df['Close'], length=5)

# 3.4 Bollinger Bands
bb = ta.bbands(df['Close'], length=14, std=2)
df['BB_Lower'] = bb['BBL_14_2.0']
df['BB_Mid'] = bb['BBM_14_2.0']
df['BB_Upper'] = bb['BBU_14_2.0']

# 3.5 Average Directional Index (ADX)
adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
df['ADX_14'] = adx['ADX_14']

# 3.6 Moving Average Convergence Divergence (MACD)
macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df['MACD'] = macd['MACD_12_26_9']
df['MACD_Signal'] = macd['MACDs_12_26_9']
df['MACD_Histogram'] = macd['MACDh_12_26_9']

# --------------------------------------------------------------------------------------
# Step 4: Visualize Data with Plots
# --------------------------------------------------------------------------------------

# 4.1 Plot Close Price and SMA_5
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price')
plt.plot(df.index, df['SMA_5'], label='5-Day SMA')
plt.title("Tesla Stock Price and 5-Period SMA (2020)")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# 4.2 Plot Close Price, SMA_5, and EMA_5
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price')
plt.plot(df.index, df['SMA_5'], label='5-Day SMA')
plt.plot(df.index, df['EMA_5'], label='5-Day EMA')
plt.title("Tesla Stock Price, SMA, and EMA (2020)")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# 4.3 Plot RSI
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['RSI_5'], label='5-Day RSI', color='orange')
plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
plt.title("Tesla Stock RSI (2020)")
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.show()

# 4.4 Plot Bollinger Bands
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price')
plt.plot(df.index, df['BB_Lower'], label='Lower Band', color='red')
plt.plot(df.index, df['BB_Mid'], label='Middle Band', color='orange')
plt.plot(df.index, df['BB_Upper'], label='Upper Band', color='red')
plt.title("Tesla Stock with Bollinger Bands (2020)")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# 4.5 Plot ADX
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ADX_14'], label='14-Day ADX', color='purple')
plt.title("Tesla Stock ADX (2020)")
plt.xlabel('Date')
plt.ylabel('ADX')
plt.legend()
plt.show()

# 4.6 Plot MACD
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['MACD'], label='MACD', color='blue')
plt.plot(df.index, df['MACD_Signal'], label='Signal Line', color='red')
plt.bar(df.index, df['MACD_Histogram'], label='Histogram', color='gray', alpha=0.5)
plt.title("Tesla Stock MACD (2020)")
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# --------------------------------------------------------------------------------------
# Step 5: Plot Regular Candlestick Chart
# --------------------------------------------------------------------------------------

fig = go.Figure(data=[go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close)])
fig.update_layout(title="Tesla Stock Price (2020)")
fig.show()

# --------------------------------------------------------------------------------------
# Step 6: End of Analysis
# --------------------------------------------------------------------------------------

