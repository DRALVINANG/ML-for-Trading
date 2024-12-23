#--------------------------------------------------------------------------------------
# Step 1: Pip Install and Import Libraries
#--------------------------------------------------------------------------------------

# 1a) Install and Import Data Manipulation Libraries
# Ensure compatible NumPy version for Pyfolio
import os
os.system("pip install numpy==1.23.0")

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import talib as ta

# Install Pyfolio
#!pip install pyfolio
import pyfolio as pf

# Install Yahoo Finance
#!pip install yfinance
import yfinance as yf

#--------------------------------------------------------------------------------------
# Step 2: Import Dataset & Plot
#--------------------------------------------------------------------------------------

ticker = 'D05.SI'
data = yf.download(ticker, start='2022-01-01')
data.columns = data.columns.droplevel(level=1)

print(data)

# Plot Close Price
data['Close'].plot(figsize=(18, 5), color='b')
plt.ylabel('Close Price')
plt.xlabel('Date')
plt.title(f'{ticker} Close Price')
plt.show()

#--------------------------------------------------------------------------------------
# Step 3: Create a Function to Define Features and Target
#--------------------------------------------------------------------------------------

def get_target_features(data):
    # Define Features (X)
    data['PCT_CHANGE'] = data['Close'].pct_change()
    data['VOLATILITY'] = data['PCT_CHANGE'].rolling(14).std() * 100
    data['SMA'] = ta.SMA(data['Close'], timeperiod=14)
    data['CORR'] = ta.CORREL(data['Close'], data['SMA'], timeperiod=14)
    data['RSI'] = ta.RSI(data['Close'].values, timeperiod=14)
    data['ADX'] = ta.ADX(data['High'].values,
                         data['Low'].values,
                         data['Close'].values,
                         timeperiod=14)

    # Define Target (y)
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)

    # Drop NaN rows
    data = data.dropna()

    return data['Actual_Signal'], data[['VOLATILITY', 'CORR', 'RSI', 'ADX']]

#--------------------------------------------------------------------------------------
# Step 4: Train Test Split
#--------------------------------------------------------------------------------------

y, X = get_target_features(data)

# Split data into training and testing
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

#--------------------------------------------------------------------------------------
# Step 5: Use StandardScaler() to Scale the X
#--------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#--------------------------------------------------------------------------------------
# Step 6: Import Logistic Regression Model and Start Training
#--------------------------------------------------------------------------------------

# Train the model
model = LogisticRegression()
model = model.fit(X_train, y_train)

# Predict and display results
y_pred = model.predict(X_test)
probability = model.predict_proba(X_test)

print("Predicted Probabilities:")
print(probability)

#--------------------------------------------------------------------------------------
# Step 7: Confusion Matrix and Accuracy Metric
#--------------------------------------------------------------------------------------

def get_metrics(y_test, predicted):
    confusion_matrix_data = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix_data, fmt="d", cmap='Blues', cbar=False, annot=True, ax=ax)
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('Actual Labels', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.xaxis.set_ticklabels(['No Position', 'Long Position'])
    ax.yaxis.set_ticklabels(['No Position', 'Long Position'])
    plt.show()

    print(metrics.classification_report(y_test, predicted))

get_metrics(y_test, y_pred)

#--------------------------------------------------------------------------------------
# Step 8: Backtesting Our Model
#--------------------------------------------------------------------------------------

ticker = 'D05.SI'
df = yf.download(ticker, start='2016-01-01', end='2017-01-01')
df.columns = df.columns.droplevel(level=1)

# Plot Close Price
df['Close'].plot(figsize=(18, 5), color='b')
plt.ylabel('Close Price')
plt.xlabel('Date')
plt.title(f'{ticker} Close Price')
plt.show()

# Define features for backtesting
df['PCT_CHANGE'] = df['Close'].pct_change()
df['VOLATILITY'] = df['PCT_CHANGE'].rolling(14).std() * 100
df['SMA'] = ta.SMA(df['Close'], 14)
df['CORR'] = ta.CORREL(df['Close'], df['SMA'], 14)
df['RSI'] = ta.RSI(df['Close'].values, timeperiod=14)
df['ADX'] = ta.ADX(df['High'].values,
                   df['Low'].values,
                   df['Open'].values, timeperiod=14)
df = df.dropna()

# Scale and predict
df_scaled = sc.transform(df[['VOLATILITY', 'CORR', 'RSI', 'ADX']])
df['predicted_signal_4_tmrw'] = model.predict(df_scaled)

# Create the strategy returns
df['strategy_returns'] = df['predicted_signal_4_tmrw'].shift(1) * df['PCT_CHANGE']
df.dropna(inplace=True)

#--------------------------------------------------------------------------------------
# Step 9: Using Pyfolio
#--------------------------------------------------------------------------------------

perf_stats = pf.timeseries.perf_stats(df.strategy_returns)

# Print the performance stats
print(perf_stats)

# Force display of Pyfolio tearsheet
import matplotlib.pyplot as plt
pf.create_simple_tear_sheet(df.strategy_returns)
plt.show()

#--------------------------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------------------------
