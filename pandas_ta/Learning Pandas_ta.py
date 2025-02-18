#---------------------------------------------------------
# Step 1: Install and Import Pandas TA
#---------------------------------------------------------
import pandas as pd
import pandas_ta as ta

df = pd.read_csv('https://gist.githubusercontent.com/DRALVINANG/5821855d6bcce977fc7f7638bb7ea9a3/raw/9d5bf33a581bf81a8319baf9d677eef309a2d7e9/TSLA%2520Stock%2520Price%2520(2020).csv')
print(df)

#---------------------------------------------------------
# Step 2: Various Types of Help with Pandas TA
#---------------------------------------------------------

# Help about this, 'ta', extension
help(df.ta)

# List of all indicators
df.ta.indicators()

# Help about an indicator such as RSI
help(ta.rsi)

#---------------------------------------------------------
# Step 3: Calculate the 5-period RSI Indicator
#---------------------------------------------------------

df['RSI_5'] = ta.rsi(close=df.Close, length=5)
print(df.head(20))

import matplotlib.pyplot as plt

plt.figure(figsize=(42, 5))
plt.plot(df.Date, df.RSI_5)
plt.xticks(rotation=90)
plt.show()

#---------------------------------------------------------
# Step 4: Calculate the 5-period Simple Moving Average (SMA)
#---------------------------------------------------------

df['SMA_5'] = ta.sma(close=df.Close, length=5)
print(df.head(20))

plt.figure(figsize=(42, 5))
plt.plot(df.Date, df.Close)
plt.plot(df.Date, df.SMA_5)
plt.xticks(rotation=90)
plt.show()

#---------------------------------------------------------
# Step 5: Plot Normal Candlestick Chart for Stock Data
#---------------------------------------------------------

import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(
                    x=df.Date,
                    open=df.Open,
                    high=df.High,
                    low=df.Low,
                    close=df.Close
                )])

fig.show()

#---------------------------------------------------------
# Step 6: Calculate Heikin Ashi Candlestick Data
#---------------------------------------------------------

help(ta.ha)

ha_df = ta.ha(open_=df.Open, high=df.High, low=df.Low, close=df.Close)
print(ha_df)

#---------------------------------------------------------
# Step 7: Plot Heikin Ashi Candlestick Chart
#---------------------------------------------------------

fig = go.Figure(data=[go.Candlestick(
                    x=df.Date,
                    open=ha_df.HA_open,
                    high=ha_df.HA_high,
                    low=ha_df.HA_low,
                    close=ha_df.HA_close
                )])

fig.show()ac

#---------------------------------------------------------
# Step 8: Calculate the 14-period ADX Indicator
#---------------------------------------------------------

a = ta.adx(df.High, df.Low, df.Close, length=14)
df = df.join(a)

df.drop(['DMP_14', 'DMN_14'], axis=1, inplace=True)

print(df)

plt.figure(figsize=(42, 5))
plt.plot(df.Date, df.ADX_14)
plt.xticks(rotation=90)
plt.show()

#---------------------------------------------------------
# Step 9: Calculate and Plot Bollinger Bands for Stock Data
#---------------------------------------------------------

bb = ta.bbands(df.Close, length=14, std=2)
bb.drop(['BBB_14_2.0', 'BBP_14_2.0'], axis=1, inplace=True)

# Renaming columns
bb.columns = ['Lower', 'Mid', 'Upper']
df = df.join(bb)

print(df)

plt.plot(df.Date, df.Close)
plt.plot(df.Date, df.Lower)
plt.plot(df.Date, df.Mid)
plt.plot(df.Date, df.Upper)
plt.xticks(rotation=90)
plt.show()

#---------------------------------------------------------
# THE END
#---------------------------------------------------------

