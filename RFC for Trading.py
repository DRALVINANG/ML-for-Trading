#--------------------------------------------------------------------------------------
# Step 1: Pip Install and Import Libraries
#--------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import pyfolio as pf
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tabulate import tabulate
from sklearn.tree import export_graphviz
from graphviz import Source
from PyPDF2 import PdfMerger

# Set display options for DataFrame
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:.5f}'.format)

#--------------------------------------------------------------------------------------
# Step 2: Import Dataset & Plot
#--------------------------------------------------------------------------------------

ticker = 'D05.SI'
data = yf.download(ticker, start='2022-01-01', end='2022-12-31')
data.columns = data.columns.droplevel(level=1)

# Round the data to 5 decimal places
data = data.round(5)

# Display data using tabulate
print(tabulate(data.head(), headers='keys', tablefmt='pretty'))

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
    # Define Features (X)
    data['PCT_CHANGE'] = data['Close'].pct_change()
    data['VOLATILITY'] = data['PCT_CHANGE'].rolling(14).std() * 100
    data['SMA'] = ta.sma(data['Close'], timeperiod=14)
    data['CORR'] = data['Close'].rolling(14).corr(data['SMA'])
    data['RSI'] = ta.rsi(data['Close'], timeperiod=14)
    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)['ADX_14']

    # Define Target (y)
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)

    # Drop NaN rows
    data = data.dropna()

    return data['Actual_Signal'], data[['VOLATILITY', 'CORR', 'RSI', 'ADX']]

#--------------------------------------------------------------------------------------
# Step 4: Train Test Split
#--------------------------------------------------------------------------------------

# Get target and features
y, X = get_target_features(data)

# Display first 10 rows of combined DataFrame with 5 decimal places
df_combined = pd.DataFrame({
    "VOLATILITY": X['VOLATILITY'],
    "CORR": X['CORR'],
    "RSI": X['RSI'],
    "ADX": X['ADX'],
    "Returns_4_Tmrw": data['Returns_4_Tmrw'],
    "Actual_Signal": y
})
df_combined = df_combined.dropna().round(5)
print(tabulate(df_combined.head(10), headers='keys', tablefmt='pretty'))

# Split data into train and test sets
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
# Step 6: Train the Random Forest Model
#--------------------------------------------------------------------------------------

# Initialize and train the model
rfc = RandomForestClassifier(n_estimators=10, criterion="entropy", max_depth=3, max_features=3)
model = rfc.fit(X_train, y_train)

#--------------------------------------------------------------------------------------
# Step 7: Visualize Random Forest Trees and Export as PDF
#--------------------------------------------------------------------------------------

merger = PdfMerger()

# Iterate over each estimator (tree) in the RandomForestClassifier
for i, estimator in enumerate(rfc.estimators_):
    dot_data = export_graphviz(
        estimator,
        out_file=None,
        filled=True,
        feature_names=X.columns,
        class_names=['No Position', 'Long Position'],
        rounded=True
    )
    graph = Source(dot_data)
    graph.format = "pdf"
    graph_binary = graph.pipe()

    # Save each tree as a PDF
    with open(f"temp_tree_{i}.pdf", "wb") as temp_file:
        temp_file.write(graph_binary)
        merger.append(temp_file.name)

# Merge all PDFs into one
output_pdf = "random_forest_trees.pdf"
merger.write(output_pdf)
merger.close()

# Remove temporary PDFs
for i in range(len(rfc.estimators_)):
    temp_pdf = f"temp_tree_{i}.pdf"
    if os.path.exists(temp_pdf):
        os.remove(temp_pdf)

# Automatically open the merged PDF
os.system(f"start {output_pdf}")

#--------------------------------------------------------------------------------------
# Step 8: Confusion Matrix and Accuracy Metric
#--------------------------------------------------------------------------------------

def get_metrics(y_test, predicted):
    confusion_matrix_data = metrics.confusion_matrix(y_test, predicted)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix_data, fmt="d", cmap='Blues', cbar=False, annot=True, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No Position', 'Long Position'])
    ax.yaxis.set_ticklabels(['No Position', 'Long Position'])
    plt.show()

    print(metrics.classification_report(y_test, predicted))

# Predict and evaluate model performance
y_pred = model.predict(X_test)
get_metrics(y_test, y_pred)

#--------------------------------------------------------------------------------------
# Step 9: Backtesting the Model
#--------------------------------------------------------------------------------------

df_backtest = yf.download(ticker, start='2023-01-01', end='2024-01-01')
df_backtest.columns = df_backtest.columns.droplevel(level=1)

# Feature creation for backtesting
df_backtest['PCT_CHANGE'] = df_backtest['Close'].pct_change()
df_backtest['VOLATILITY'] = df_backtest['PCT_CHANGE'].rolling(14).std() * 100
df_backtest['RSI'] = ta.rsi(df_backtest['Close'], timeperiod=14)
df_backtest['ADX'] = ta.adx(df_backtest['High'], df_backtest['Low'], df_backtest['Close'], length=14)['ADX_14']
df_backtest['SMA'] = ta.sma(df_backtest['Close'], timeperiod=14)
df_backtest['CORR'] = df_backtest['Close'].rolling(window=14).corr(df_backtest['SMA'])

# Drop NaN rows and round to 5 decimal places
df_backtest = df_backtest.dropna().round(5)

# Scale features for backtesting
df_scaled_backtest = sc.transform(df_backtest[['VOLATILITY', 'CORR', 'RSI', 'ADX']])
df_backtest['predicted_signal_4_tmrw'] = model.predict(df_scaled_backtest)

# Calculate strategy returns (corrected as per your suggestion)
df_backtest['strategy_returns'] = df_backtest['predicted_signal_4_tmrw'] * df_backtest['PCT_CHANGE'].shift(-1)
df_backtest.dropna(inplace=True)

# Display backtest data
print(tabulate(df_backtest.head(10), headers='keys', tablefmt='pretty'))

#--------------------------------------------------------------------------------------
# Step 10: Using Pyfolio for Evaluation
#--------------------------------------------------------------------------------------

# Performance statistics using Pyfolio
perf_stats = pf.timeseries.perf_stats(df_backtest.strategy_returns)
print(perf_stats)

# Plot performance tear sheet
pf.create_simple_tear_sheet(df_backtest.strategy_returns)
plt.show()

