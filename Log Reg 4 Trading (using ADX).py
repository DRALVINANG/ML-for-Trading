#--------------------------------------------------------------------------------------
# Step 1: Pip Install and Import Libraries
#--------------------------------------------------------------------------------------

import os
#os.system("pip install numpy==1.23.0")
#os.system("pip install pandas==1.3.5")
#os.system("pip install tabulate")

import numpy as np
import pandas as pd
import pandas_ta as ta
import pyfolio as pf
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

pd.set_option('display.max_columns', 20)

#--------------------------------------------------------------------------------------
# Step 2: Import Dataset & Plot
#--------------------------------------------------------------------------------------

ticker = 'D05.SI'
data = yf.download(ticker, start='2022-01-01', end = '2022-12-31')
data.columns = data.columns.droplevel(level=1)

print(tabulate(data.head(), headers='keys', tablefmt='pretty'))  # Display the first few rows in a tabular format

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

# Create DataFrame of X and y to display first 10 rows
df_combined = pd.DataFrame({
    "VOLATILITY": X['VOLATILITY'],
    "CORR": X['CORR'],
    "RSI": X['RSI'],
    "ADX": X['ADX'],
    "Returns_4_Tmrw": data['Returns_4_Tmrw'],  # Added Returns_4_Tmrw here
    "Actual_Signal": y
})

df_combined = df_combined.dropna()
print(tabulate(df_combined.head(10), headers='keys', tablefmt='pretty'))  # Displaying the first 10 rows

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
# Step 6: Train the Logistic Regression Model
#--------------------------------------------------------------------------------------

# Initialize and train the model
model = LogisticRegression()
model = model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Create a DataFrame with predicted vs actual class and additional columns
test_dates = data.index[split:len(y_pred) + split]  # Match the test set size with y_pred
returns_4_tmrw = data['Returns_4_Tmrw'].iloc[split:split+len(y_pred)]  # Ensure it matches the length

# Create DataFrame
data1 = pd.DataFrame({
    "Predicted_Class": y_pred,
    "Actual_Class": y_test.tolist(),
    "Date": test_dates,  # Dates corresponding to the test set
    "Returns_4_Tmrw": returns_4_tmrw  # Corresponding returns
})

# Print the DataFrame
print(tabulate(data1.head(10), headers='keys', tablefmt='pretty'))  # Displaying the first 10 rows

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
df = yf.download(ticker, start='2023-01-01', end='2024-01-01')
df.columns = df.columns.droplevel(level=1)

# Calculate Features
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

# Scale and Predict
df_scaled = sc.transform(df[['VOLATILITY', 'CORR', 'RSI', 'ADX']])  # Using only these four features
df['predicted_signal_4_tmrw'] = model.predict(df_scaled)

# Calculate Strategy Returns
df['strategy_returns'] = df['predicted_signal_4_tmrw'].shift(1) * df['PCT_CHANGE']
df.dropna(inplace=True)

# Display the backtest data
print(tabulate(df.head(10), headers='keys', tablefmt='pretty'))  # Displaying the first 10 rows

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

