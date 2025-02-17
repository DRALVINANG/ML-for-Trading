import os
os.system("pip install numpy==1.23.0")
os.system("pip install pandas==1.3.5")

# -------------------------------------------------------------------------------------
# Step 1: Pip Install and Import Libraries
# -------------------------------------------------------------------------------------
# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import pyfolio as pf
import yfinance as yf
from graphviz import Source
from io import BytesIO

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
    data['VOLATILITY'] = data['PCT_CHANGE'].rolling(14).std() * 100
    data['RSI'] = ta.rsi(data['Close'], length=14)  # RSI using pandas_ta

    # Define Target (y)
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)

    data = data.dropna()
    return data['Actual_Signal'], data[['VOLATILITY', 'RSI']]

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
# Define and Train the Model using Entropy
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model = model.fit(X_train, y_train)

# Visualize the Decision Tree without Saving PNG
dot_data = tree.export_graphviz(model, out_file=None, 
                                filled=True, 
                                feature_names=X.columns,
                                class_names=['No Position', 'Long Position'])

# Display the Decision Tree as a PDF (No PNG Saved)
graph = Source(dot_data)
pdf_bytes = BytesIO()  # Create a BytesIO stream for PDF
graph.format = "pdf"
graph.render("decision_tree_temp", format="pdf", cleanup=True)  # Render PDF
graph.view()  # Open the PDF directly

# Predict
y_pred = model.predict(X_test)

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

get_metrics(y_test, y_pred)

# -------------------------------------------------------------------------------------
# Step 8: Backtesting the Model
# -------------------------------------------------------------------------------------
df = yf.download(ticker, start='2016-01-01', end='2017-01-01')
df.columns = df.columns.droplevel(level=1)

# Feature Creation
df['PCT_CHANGE'] = df['Close'].pct_change()
df['VOLATILITY'] = df['PCT_CHANGE'].rolling(14).std() * 100
df['RSI'] = ta.rsi(df['Close'], length=14)  # RSI using pandas_ta

df = df.dropna()

# Scale Features
df_scaled = sc.transform(df[['VOLATILITY', 'RSI']])

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

