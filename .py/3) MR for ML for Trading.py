#--------------------------------------------------------------------------------------
# Step 1: Pip Install and Import Libraries
#--------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import talib as ta
import yfinance as yf
import pyfolio as pf

# Ensure compatible NumPy version for Pyfolio
import os
os.system("pip install numpy==1.23.0")
os.system("pip install pandas==1.3.5")

# -------------------------------------------------------------------------------------
# Step 2: Import Dataset & Plot
# -------------------------------------------------------------------------------------

ticker = 'D05.SI'
data = yf.download(ticker, start='2022-01-01')
data.columns = data.columns.droplevel(level=1)

# Display the dataset
print(data.head())

# Plot the Close Price
data['Close'].plot(figsize=(18, 5), color='b')
plt.ylabel('Close Price')
plt.xlabel('Date')
plt.title(f'{ticker} Close Price')
plt.show()

# -------------------------------------------------------------------------------------
# Step 3: Define Features and Target
# -------------------------------------------------------------------------------------

def get_target_features(data):
    # Define Features (X)
    data['PCT_CHANGE'] = data['Close'].pct_change()
    data['VOLATILITY'] = data['PCT_CHANGE'].rolling(14).std() * 100
    data['SMA'] = ta.SMA(data['Close'], timeperiod=14)
    data['CORR'] = ta.CORREL(data['Close'], data['SMA'], timeperiod=14)
    data['RSI'] = ta.RSI(data['Close'].values, timeperiod=14)
    data['ADX'] = ta.ADX(data['High'].values, data['Low'].values, data['Open'].values, timeperiod=14)

    # Define Target (y)
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)

    data = data.dropna()
    return data['Actual_Signal'], data[['VOLATILITY', 'CORR', 'RSI', 'ADX']]

# -------------------------------------------------------------------------------------
# Step 4: Train-Test Split
# -------------------------------------------------------------------------------------

y, X = get_target_features(data)

split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# -------------------------------------------------------------------------------------
# Step 5: Scaling
# -------------------------------------------------------------------------------------

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -------------------------------------------------------------------------------------
# Step 6: Define, Train the Model, and Predict
# -------------------------------------------------------------------------------------

# Define and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the Model
y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

# -------------------------------------------------------------------------------------
# Step 7: Confusion Matrix and Accuracy Metric
# -------------------------------------------------------------------------------------

def get_metrics(y_test, predicted):
    confusion_matrix_data = metrics.confusion_matrix(y_test, predicted)

    # Plot the Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix_data, fmt="d", cmap='Blues', cbar=False, annot=True, ax=ax)
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('Actual Labels', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.xaxis.set_ticklabels(['No Position', 'Long Position'])
    ax.yaxis.set_ticklabels(['No Position', 'Long Position'])
    plt.show()

    # Print Classification Report
    print(metrics.classification_report(y_test, predicted))

get_metrics(y_test, y_pred)

# -------------------------------------------------------------------------------------
# Step 8: Backtesting Our Model
# -------------------------------------------------------------------------------------

df = yf.download(ticker, start='2016-01-01', end='2017-01-01')
df.columns = df.columns.droplevel(level=1)


df['PCT_CHANGE'] = df['Close'].pct_change()
df['VOLATILITY'] = df['PCT_CHANGE'].rolling(14).std() * 100
df['SMA'] = ta.SMA(df['Close'], timeperiod=14)
df['CORR'] = ta.CORREL(df['Close'], df['SMA'], timeperiod=14)
df['RSI'] = ta.RSI(df['Close'].values, timeperiod=14)
df['ADX'] = ta.ADX(df['High'].values,
                   df['Low'].values,
                   df['Open'].values,
                   timeperiod=14)

df = df.dropna()

df_scaled = sc.transform(df[['VOLATILITY', 'CORR', 'RSI', 'ADX']])
df['predicted_signal_4_tmrw'] = model.predict(df_scaled)

df['strategy_returns'] = df['predicted_signal_4_tmrw'].shift(1) * df['PCT_CHANGE']
df.dropna(inplace=True)

# -------------------------------------------------------------------------------------
# Step 9: Using Pyfolio
# -------------------------------------------------------------------------------------

# Force display of Pyfolio tearsheet
import matplotlib.pyplot as plt

perf_stats = pf.timeseries.perf_stats(df.strategy_returns)

# Print the performance stats
print(perf_stats)

pf.create_simple_tear_sheet(df.strategy_returns)
plt.show()

# -------------------------------------------------------------------------------------
# THE END
# -------------------------------------------------------------------------------------

