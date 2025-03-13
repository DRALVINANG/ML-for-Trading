# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------
# Step 2: Download Data
# -----------------------------
# Download data for TSLA (Tesla) from Yahoo Finance
ticker = 'TSLA'
start_date = '2020-01-01'
end_date = '2022-01-01'
df = yf.download(ticker, start=start_date, end=end_date)

# Flatten multi-level columns (in case the update to yfinance requires it)
df.columns = df.columns.droplevel(level=1)

# -----------------------------
# Step 3: Plot the RSI (Relative Strength Index)
# -----------------------------
# Calculate RSI
df['RSI'] = ta.rsi(df['Close'], length=14)

# Plot the RSI
df['RSI'].plot()
plt.title('RSI (Relative Strength Index) - TSLA')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.show()

# -----------------------------
# Step 4: Create the Signal using RSI
# -----------------------------
# Create Buy (1) and Sell (0) signals based on RSI thresholds
df['Signal'] = np.where(df['RSI'] < 30, 1,           # 1 means BUY
                        np.where(df['RSI'] > 70, 0,  # 0 means SELL
                                 np.nan))

# Drop rows with NaN values
df1 = df[['RSI', 'Signal']].dropna()

# Print the signal DataFrame
print(df1)

# -----------------------------
# Step 5: Train-Test Split
# -----------------------------
# Split the data into training and testing sets
X = df1.RSI
y = df1.Signal

split = int(0.8 * len(X))

X_train, X_test = X[:split].values.reshape(-1, 1), X[split:].values.reshape(-1, 1)
y_train, y_test = y[:split], y[split:]

# -----------------------------
# Step 6: Fit Logistic Regression Model & Get Coefficient and Intercept
# -----------------------------
# Initialize and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get the coefficient and intercept
a = model.coef_
b = model.intercept_

# Print model coefficient and intercept
print(f'Model Coefficient: {a}')
print(f'Model Intercept: {b}')

# -----------------------------
# Step 7: Calculate Midpoint (50% point)
# -----------------------------
# Calculate midpoint where logistic function = 50%
midpoint = -b / a
print(f'Midpoint RSI (50% probability): {midpoint[0][0]}')

# -----------------------------
# Step 8: Plot the Logistic Regression Curve with Midpoint
# -----------------------------
# Define logistic function
def logistic_func(X, a, b):
    return 1 / (1 + np.exp(-(a * X + b)))

# Generate x-axis values for the smooth logistic curve
X_new = np.linspace(min(X), max(X), 100)

# Calculate y-values for the logistic function
y_new = logistic_func(X_new, a, b).flatten()

# Plot the data points and the logistic regression curve
plt.scatter(X, y, label='Data Points')
plt.plot(X_new, y_new, color='red', label='Logistic Curve')

# Add vertical line at midpoint
plt.axvline(midpoint[0][0], color='green', linestyle='--', label=f'Midpoint RSI = {midpoint[0][0]:.2f}')

# Add horizontal line at y=0.5 (50% probability level)
plt.axhline(0.5, color='blue', linestyle='--', label='50% Probability')

# Customize the plot for better visualization
plt.title('Logistic Regression Curve with Midpoint')
plt.xlabel('RSI')
plt.ylabel('Probability (BUY/SELL)')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Step 9: Compare Predicted vs Actual
# -----------------------------
# Make predictions on the test set
y_pred = model.predict(X_test)

# Create a comparison DataFrame
comparison = pd.DataFrame({
    'Predicted': y_pred,
    'Actual': y_test
})

print(comparison)

# -----------------------------
# Step 10: Check the Model Accuracy
# -----------------------------
# Evaluate the model's performance using classification report
# Dynamically get the labels present in the predicted data
labels = np.unique(y_pred)

# Adjust target_names according to the present labels
target_names = [f'class {int(label)}: {"BUY" if label == 1 else "SELL"}' for label in labels]

print(classification_report(y_test, y_pred, target_names=target_names, labels=labels))

# -----------------------------
# Step 11: Predict Example
# -----------------------------
# Predict the signal for an RSI value of 55
test = model.predict([[55]])
print(f'Predicted signal for RSI 55: {test[0]}')

# If RSI < 30 --> Buy (1)
# If RSI > 70 --> Sell (0)
# Based on the logistic curve, anything below a certain threshold will be BUY
