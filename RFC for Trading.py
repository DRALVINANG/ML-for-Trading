# -------------------------------------------------------------------------------------
# Step 1: Install Libraries
# -------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas_ta as ta  # Use pandas_ta for technical indicators
import pyfolio as pf
import yfinance as yf
from graphviz import Source
from sklearn.tree import export_graphviz
from PyPDF2 import PdfMerger
import os

# -------------------------------------------------------------------------------------
# Step 2: Import Dataset and Plot
# -------------------------------------------------------------------------------------

ticker = 'D05.SI'
data = yf.download(ticker, start='2022-01-01', end = '2023-01-01')
data.columns = data.columns.droplevel(level=1)

# Plot Close Price
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
    data['SMA'] = ta.sma(data['Close'], length=14)
    data['CORR'] = data['Close'].rolling(14).corr(data['SMA'])
    data['RSI'] = ta.rsi(data['Close'], length=14)  # RSI using pandas_ta
    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)['ADX_14']

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

# Define and Train Random Forest Model
rfc = RandomForestClassifier(n_estimators=10, criterion="entropy", max_depth=3, max_features=3)
model = rfc.fit(X_train, y_train)

# -------------------------------------------------------------------------------------
# Visualize Random Forest Trees (Consolidated into a Single PDF)
# -------------------------------------------------------------------------------------

merger = PdfMerger()  # Initialize PdfMerger to combine PDFs

# Iterate over estimators in the Random Forest
for i, estimator in enumerate(rfc.estimators_):
    # Export each tree as a DOT file
    dot_data = export_graphviz(
        estimator,
        out_file=None,
        filled=True,
        feature_names=X.columns,
        class_names=['No Position', 'Long Position'],
        rounded=True
    )

    # Render each tree as a PDF
    graph = Source(dot_data)
    graph.format = "pdf"
    graph_binary = graph.pipe()

    # Add rendered PDF directly to the merger
    with open(f"temp_tree_{i}.pdf", "wb") as temp_file:
        temp_file.write(graph_binary)
        merger.append(temp_file.name)

# Consolidate all PDFs into a single PDF
output_pdf = "random_forest_trees.pdf"
merger.write(output_pdf)  # Save combined PDF
merger.close()

# Remove temporary files
for i in range(len(rfc.estimators_)):
    temp_pdf = f"temp_tree_{i}.pdf"
    if os.path.exists(temp_pdf):
        os.remove(temp_pdf)

# Open the consolidated PDF
os.system(f"start {output_pdf}")  # Automatically open the PDF

# -------------------------------------------------------------------------------------
# Step 7: Confusion Matrix and Accuracy Metrics
# -------------------------------------------------------------------------------------

def get_metrics(y_test, predicted):
    confusion_matrix_data = metrics.confusion_matrix(y_test, predicted)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix_data, fmt="d", cmap='Blues', cbar=False, annot=True, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No Position', 'Long Position'])
    ax.yaxis.set_ticklabels(['No Position', 'Long Position'])
    plt.show()

    print(metrics.classification_report(y_test, predicted))

# Predict
y_pred = model.predict(X_test)

# Get confusion matrix and classification report
get_metrics(y_test, y_pred)

# -------------------------------------------------------------------------------------
# Step 8: Backtesting the Model
# -------------------------------------------------------------------------------------

df = yf.download(ticker, start='2023-01-01', end='2024-01-01')
df.columns = df.columns.droplevel(level=1)

# Feature Creation
df['PCT_CHANGE'] = df['Close'].pct_change()
df['VOLATILITY'] = df['PCT_CHANGE'].rolling(14).std() * 100
df['SMA'] = ta.sma(df['Close'], length=14)
df['CORR'] = df['Close'].rolling(14).corr(df['SMA'])
df['RSI'] = ta.rsi(df['Close'], length=14)  # RSI using pandas_ta
df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']

df = df.dropna()

# Scale Features
df_scaled = sc.transform(df[['VOLATILITY', 'CORR', 'RSI', 'ADX']])

# Predict and Calculate Strategy Returns
df['predicted_signal_4_tmrw'] = model.predict(df_scaled)
df['strategy_returns'] = df['predicted_signal_4_tmrw'].shift(1) * df['PCT_CHANGE']
df.dropna(inplace=True)

# -------------------------------------------------------------------------------------
# Step 9: Evaluate with Pyfolio
# -------------------------------------------------------------------------------------

perf_stats = pf.timeseries.perf_stats(df.strategy_returns)

# Print the performance stats
print(perf_stats)

pf.create_simple_tear_sheet(df.strategy_returns)
plt.show()

# -------------------------------------------------------------------------------------
# THE END
# -------------------------------------------------------------------------------------

