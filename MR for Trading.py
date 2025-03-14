#--------------------------------------------------------------------------------------
# Step 1: Pip Install and Import Libraries
#--------------------------------------------------------------------------------------

import os
os.system("pip install numpy==1.23.0")
os.system("pip install pandas==1.3.5")

import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import pyfolio as pf
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression  # Changed to Multiple Linear Regression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#--------------------------------------------------------------------------------------
# Step 2: Import Dataset & Plot
#--------------------------------------------------------------------------------------

ticker = 'D05.SI'
data = yf.download(ticker, start='2020-01-01', end='2021-01-01')
data.columns = data.columns.droplevel(level=1)

# Print the first few rows of the data
print(data.head())

# Plotting the Close Price
data['Close'].plot(figsize=(18, 5), color='b')
plt.ylabel('Close Price')
plt.xlabel('Date')
plt.title(f'{ticker} Close Price')
plt.show()

#--------------------------------------------------------------------------------------
# Step 3: Create a Function to Define Features and Target
#--------------------------------------------------------------------------------------

def get_target_features(data):
    # -------------------------------------------------------------
    # Define Features (X)
    
    # Volatility (Using Percentage Change for volatility calculation)
    data['PCT_CHANGE'] = data['Close'].pct_change()
    data['VOLATILITY'] = data['PCT_CHANGE'].rolling(14).std() * 100

    # RSI (Relative Strength Index)
    data['RSI'] = ta.rsi(data['Close'], timeperiod=14)

    # ADX (Average Directional Index) and its components
    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)['ADX_14']

    # Correlation: Correlation between 'Close' and its 14-day moving average (SMA)
    data['SMA'] = ta.sma(data['Close'], timeperiod=14)
    data['CORR'] = data['Close'].rolling(window=14).corr(data['SMA'])

    # ------------------------------------------------------------
    # Define Target (y)
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)

    # Create the signal column (1 = BUY, 0 = DO NOT BUY)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)

    # Drop NaN rows (necessary after rolling and shifting operations)
    data = data.dropna()

    return data['Actual_Signal'], data[['VOLATILITY', 'CORR', 'RSI', 'ADX']]

#--------------------------------------------------------------------------------------
# Step 4: Train Test Split
#--------------------------------------------------------------------------------------

# Split Data
y, X = get_target_features(data)

# Split into 80% training and 20% testing
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

#--------------------------------------------------------------------------------------
# Step 5: Scale the Features
#--------------------------------------------------------------------------------------

# Initialize StandardScaler
sc = StandardScaler()

# Scale X_train and X_test
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#--------------------------------------------------------------------------------------
# Step 6: Train the Multiple Linear Regression Model
#--------------------------------------------------------------------------------------

# Initialize and train the model
model = LinearRegression()  # Multiple Linear Regression
model = model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Since this is regression, let's categorize the continuous values into binary 0 or 1
y_pred = np.where(y_pred > 0.5, 1, 0)

# Show the predicted values
print(f'Predicted values: {y_pred}')

#--------------------------------------------------------------------------------------
# Step 7: Confusion Matrix and Accuracy Metric
#--------------------------------------------------------------------------------------

def get_metrics(y_test, predicted):
    confusion_matrix_data = metrics.confusion_matrix(y_test, predicted)
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix_data, fmt="d", cmap='Blues', cbar=False, annot=True, ax=ax)

    # Set axes labels and title
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('Actual Labels', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.xaxis.set_ticklabels(['No Position', 'Long Position'])
    ax.yaxis.set_ticklabels(['No Position', 'Long Position'])

    # Display the plot
    plt.show()

    print('\n\n\n', metrics.classification_report(y_test, predicted))

# Get confusion matrix and classification report
get_metrics(y_test, y_pred)

#--------------------------------------------------------------------------------------
# Step 8: Backtesting Our Model
#--------------------------------------------------------------------------------------

# Get new data for backtesting
ticker = 'D05.SI'
df = yf.download(ticker, start='2021-01-01', end='2022-01-01')
df.columns = df.columns.droplevel(level=1)

# Calculate Features for backtesting data
df['PCT_CHANGE'] = df['Close'].pct_change()
df['VOLATILITY'] = df['PCT_CHANGE'].rolling(14).std() * 100

# Calculate RSI for the backtesting data
df['RSI'] = ta.rsi(df['Close'], timeperiod=14)

# Calculate ADX for the backtesting data
df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']

# Correlation
df['SMA'] = ta.sma(df['Close'], timeperiod=14)
df['CORR'] = df['Close'].rolling(window=14).corr(df['SMA'])

df = df.dropna()

# Scale and Predict on backtesting data
df_scaled = sc.transform(df[['VOLATILITY', 'CORR', 'RSI', 'ADX']])  # Using only these four features
df['predicted_signal_4_tmrw'] = model.predict(df_scaled)

# Calculate Strategy Returns
df['strategy_returns'] = df['predicted_signal_4_tmrw'].shift(1) * df['PCT_CHANGE']
df.dropna(inplace=True)

#--------------------------------------------------------------------------------------
# Step 9: Using Pyfolio
#--------------------------------------------------------------------------------------

# Performance statistics using Pyfolio
perf_stats = pf.timeseries.perf_stats(df.strategy_returns)

# Print the performance stats
print(perf_stats)

# Force display of Pyfolio tearsheet
import matplotlib.pyplot as plt
pf.create_simple_tear_sheet(df.strategy_returns)
plt.show()

#--------------------------------------------------------------------------------------
# End of the Script
#--------------------------------------------------------------------------------------

