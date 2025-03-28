import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import pyfolio as pf
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tabulate import tabulate

# -------------------------------------------------------------------------------------
# Step 1: Install Libraries (for completeness, but you have them commented out)
# -------------------------------------------------------------------------------------

# os.system("pip install numpy==1.23.0")
# os.system("pip install pandas==1.3.5")

# -------------------------------------------------------------------------------------
# Step 2: Import Dataset and Plot
# -------------------------------------------------------------------------------------

ticker = 'D05.SI'
data = yf.download(ticker, start='2022-01-01', end='2023-01-01')
data.columns = data.columns.droplevel(level=1)

# Display the dataset using tabulate
print(tabulate(data.head(), headers='keys', tablefmt='pretty'))

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
    data['VOLATILITY'] = data['PCT_CHANGE'].rolling(14).std() * 100
    data['RSI'] = ta.rsi(data['Close'], length=14)  # RSI using pandas_ta
    adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)  # ADX using pandas_ta
    data['ADX'] = adx['ADX_14']  # Extract ADX from the result
    data['SMA'] = ta.sma(data['Close'], length=14)  # Simple Moving Average
    
    # Correlation: Correlation between 'Close' and its 14-day moving average (SMA)
    data['CORR'] = data['Close'].rolling(window=14).corr(data['SMA'])  # Correlation using pandas_ta

    # Define Target (y)
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)

    data = data.dropna()
    return data['Actual_Signal'], data[['VOLATILITY', 'RSI', 'ADX', 'CORR']]

y, X = get_target_features(data)

# Display Features and Target using tabulate
df_combined = pd.DataFrame({
    "VOLATILITY": X['VOLATILITY'],
    "RSI": X['RSI'],
    "ADX": X['ADX'],
    "CORR": X['CORR'],
    "Returns_4_Tmrw": data['Returns_4_Tmrw'],  # Added Returns_4_Tmrw here
    "Actual_Signal": y
})
df_combined = df_combined.dropna().round(5)  # Round to 5 decimal places
print(tabulate(df_combined.head(10), headers='keys', tablefmt='pretty'))

# -------------------------------------------------------------------------------------
# Step 4: Train Test Split
# -------------------------------------------------------------------------------------

# Split Data
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# -------------------------------------------------------------------------------------
# Step 5: Scaling
# -------------------------------------------------------------------------------------

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -------------------------------------------------------------------------------------
# Step 6: Define and Train the Model (ANN)
# -------------------------------------------------------------------------------------

model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=X.shape[1]))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

# -------------------------------------------------------------------------------------
# Step 7: Plot the Model Architecture
# -------------------------------------------------------------------------------------

keras.utils.plot_model(model, 'model.png', show_shapes=True)

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Train Model
history = model.fit(X_train, y_train, batch_size=20, epochs=100, validation_data=(X_test, y_test))

# -------------------------------------------------------------------------------------
# Step 8: Plot Training Metrics
# -------------------------------------------------------------------------------------

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
# Step 9: Confusion Matrix and Accuracy Metrics
# -------------------------------------------------------------------------------------

from sklearn import metrics

def get_metrics(y_test, predicted):
    confusion_matrix_data = metrics.confusion_matrix(y_test, predicted)
    print(tabulate(confusion_matrix_data, headers=['Predicted No Position', 'Predicted Long Position'], tablefmt='pretty'))
    print('\n\n\n', metrics.classification_report(y_test, predicted))

y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
get_metrics(y_test, y_pred)

# -------------------------------------------------------------------------------------
# Step 10: Backtesting Our Model
# -------------------------------------------------------------------------------------

df = yf.download(ticker, start='2023-01-01', end='2024-01-01')
df.columns = df.columns.droplevel(level=1)

df['PCT_CHANGE'] = df['Close'].pct_change()
df['VOLATILITY'] = df['PCT_CHANGE'].rolling(14).std() * 100
df['RSI'] = ta.rsi(df['Close'], length=14)  # RSI using pandas_ta
df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']  # ADX using pandas_ta
df['SMA'] = ta.sma(df['Close'], length=14)  # Simple Moving Average
df['CORR'] = df['Close'].rolling(window=14).corr(df['SMA'])  # Correlation using pandas_ta

df.dropna(inplace=True)
df_scaled = sc.transform(df[['VOLATILITY', 'RSI', 'ADX', 'CORR']])
df['predicted_signal_4_tmrw'] = model.predict(df_scaled)
df['strategy_returns'] = df['predicted_signal_4_tmrw'].shift(1) * df['PCT_CHANGE']
df.dropna(inplace=True)

# Display the backtest data using tabulate
print(tabulate(df.head(10), headers='keys', tablefmt='pretty'))

# -------------------------------------------------------------------------------------
# Step 11: Using Pyfolio
# -------------------------------------------------------------------------------------

perf_stats = pf.timeseries.perf_stats(df.strategy_returns)

# Print the performance stats
print(perf_stats)

pf.create_simple_tear_sheet(df.strategy_returns)
plt.show()

# -------------------------------------------------------------------------------------
# THE END
# -------------------------------------------------------------------------------------

