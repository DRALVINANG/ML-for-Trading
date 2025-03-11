# -------------------------------------------------------------------------------------
# Step 1: Pip Install, Import Libraries, and Dataset
# -------------------------------------------------------------------------------------
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Download data for D05.SI (example ticker) from Yahoo Finance
ticker = 'D05.SI'
start_date = '2020-01-01'
data1 = yf.download(ticker, start=start_date)

# Flatten the multi-level columns in the DataFrame
data1.columns = data1.columns.droplevel(level=1)

# Visual inspection of the data (optional)
print(data1.head())

# -------------------------------------------------------------------------------------
# Step 2: Visual Inspection of Linear Relationship between Features (Pairplot)
# -------------------------------------------------------------------------------------

# Plot pairplot to visualize the relationships between columns
sns.pairplot(data1)

# Show the plot
plt.show()

# -------------------------------------------------------------------------------------
# Step 3: Define X and y (using yesterday's Open, High, Low to predict today's Close)
# -------------------------------------------------------------------------------------

# Shift the features (Open, High, Low) by 1 to predict today's Close price
X = data1[['Open', 'High', 'Low']].shift(+1)

# Drop rows with NaN values resulting from the shift
X = X.dropna()

# Target variable is the Close price
y = data1['Close']

# Combine X and y into a single DataFrame for model fitting
df1 = X.join(y)

# Ensure that both X and y are aligned before fitting the model
X = df1[['Open', 'High', 'Low']]
y = df1['Close']

# -------------------------------------------------------------------------------------
# Step 4: Train-Test Split
# -------------------------------------------------------------------------------------

# Split data into training and test sets (80% training, 20% testing)
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# -------------------------------------------------------------------------------------
# Step 5: Fit Multiple Regression Model
# -------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression

# Create the Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Use the model to make predictions on the test set
y_pred = model.predict(X_test)

# Comparing predicted vs actual values
comparison = pd.DataFrame({
    'predicted': y_pred.tolist(),
    'actual': y_test.values.tolist()
})

# Print the comparison
print(comparison)

# -------------------------------------------------------------------------------------
# Step 6: R2 Score
# -------------------------------------------------------------------------------------

from sklearn.metrics import r2_score

# Calculate R2 score for the model
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.4f}")

# -------------------------------------------------------------------------------------
# Step 7: Prediction Example
# -------------------------------------------------------------------------------------

# Test the model by predicting with a fictitious "High" price of $200
test = model.predict([[200, 200, 200]])
print(f"Predicted close price for tomorrow: ${test[0]:.2f}")

# In other words, if today's High is $200, the model predicts tomorrow's closing price will be ~$196.

