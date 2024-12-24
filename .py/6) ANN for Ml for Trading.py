# -------------------------------------------------------------------------------------
# Step 1: Install Libraries
# -------------------------------------------------------------------------------------

# Ensure compatible NumPy version for Pyfolio
import os
os.system("pip install numpy==1.23.0")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import talib as ta
import pyfolio as pf
import yfinance as yf


# -------------------------------------------------------------------------------------
# Step 2: Import Dataset and Plot
# -------------------------------------------------------------------------------------

ticker = 'D05.SI'
data = yf.download(ticker, start='2022-01-01')
data.columns = data.columns.droplevel(level=1)

print(data)

# Plot Close Price
data.Close.plot(figsize=(18, 5), color='b')
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
    data['VOLATILITY'] = data.rolling(14)['PCT_CHANGE'].std() * 100
    data['SMA'] = ta.SMA(data['Close'], 14)
    data['CORR'] = ta.CORREL(data['Close'], data['SMA'], 14)
    data['RSI'] = ta.RSI(data['Close'].values, timeperiod=14)
    data['ADX'] = ta.ADX(data['High'].values, data['Low'].values, data['Open'].values, timeperiod=14)
    
    # Define Target (y)
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)
    data = data.dropna()

    return data['Actual_Signal'], data[['VOLATILITY', 'CORR', 'RSI', 'ADX']]

y, X = get_target_features(data)

# -------------------------------------------------------------------------------------
# Step 4: Train Test Split
# -------------------------------------------------------------------------------------

split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# -------------------------------------------------------------------------------------
# Step 5: Scaling
# -------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -------------------------------------------------------------------------------------
# Step 6: Define and Train the Model
# -------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=X.shape[1]))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

# Plot Model
import pydot
import tensorflow as tf
from tensorflow import keras

keras.utils.plot_model(model, 'model.png', show_shapes=True)

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Train Model
history = model.fit(X_train, y_train, batch_size=20, epochs=100, validation_data=(X_test, y_test))

# Plot Training Metrics
bce = history.history['loss']
val_bce = history.history['val_loss']
epoch = range(len(bce))

plt.plot(epoch, bce, label='Binary Crossentropy')
plt.plot(epoch, val_bce, label='Validation Binary Crossentropy')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy')
plt.legend()

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
plt.plot(epoch, acc, label='Binary Accuracy')
plt.plot(epoch, val_acc, label='Validation Binary Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -------------------------------------------------------------------------------------
# Step 7: Confusion Matrix and Accuracy Metric
# -------------------------------------------------------------------------------------

from sklearn import metrics

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

y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
get_metrics(y_test, y_pred)

# -------------------------------------------------------------------------------------
# Step 8: Backtesting Our Model
# -------------------------------------------------------------------------------------

ticker = 'D05.SI'
df = yf.download(ticker, start='2016-01-01', end='2017-01-01')
df.columns = df.columns.droplevel(level=1)

df['PCT_CHANGE'] = df['Close'].pct_change()
df['VOLATILITY'] = df.rolling(14)['PCT_CHANGE'].std() * 100
df['SMA'] = ta.SMA(df['Close'], 14)
df['CORR'] = ta.CORREL(df['Close'], df['SMA'], 14)
df['RSI'] = ta.RSI(df['Close'].values, timeperiod=14)
df['ADX'] = ta.ADX(df['High'].values, df['Low'].values, df['Open'].values, timeperiod=14)

df.dropna(inplace=True)
df_scaled = sc.transform(df[['VOLATILITY', 'CORR', 'RSI', 'ADX']])
df['predicted_signal_4_tmrw'] = model.predict(df_scaled)
df['strategy_returns'] = df['predicted_signal_4_tmrw'].shift(1) * df['PCT_CHANGE']
df.dropna(inplace=True)

# -------------------------------------------------------------------------------------
# Step 9: Using Pyfolio
# -------------------------------------------------------------------------------------
perf_stats = pf.timeseries.perf_stats(df.strategy_returns)

# Print the performance stats
print(perf_stats)

pf.create_simple_tear_sheet(df.strategy_returns)
plt.show()

# -------------------------------------------------------------------------------------
# THE END
# -------------------------------------------------------------------------------------

