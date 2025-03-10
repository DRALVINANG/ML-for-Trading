# -------------------------------------------------------------------------------------
# Step 1: Install Libraries
# -------------------------------------------------------------------------------------
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -------------------------------------------------------------------------------------
# Step 2: Import Dataset and Plot
# -------------------------------------------------------------------------------------

# Import Data from 2020 to 2021
ticker = 'TSLA'
df = yf.download(ticker, start='2020-01-01', end='2021-12-31')

# Plotting the Close Price
df1 = df.reset_index()
df1 = df1[['Date', 'Close']]
df1.set_index('Date', inplace=True)

plt.style.use('ggplot')
plt.figure(figsize=(22, 5))
plt.plot(df1, 'b')
plt.plot(df1, 'ro')
plt.grid(True)
plt.title('TESLA Close Price Representation')
plt.xlabel('Trading Days')
plt.xticks(rotation=90)
plt.ylabel('TESLA Close Price')
plt.show()

# -------------------------------------------------------------------------------------
# Step 3: Identifying a Linear Relationship between High and Close Price
# -------------------------------------------------------------------------------------

# Plot Scatter Plot of High vs Close Price
plt.figure(figsize=(12, 6))
plt.scatter(df['High'], df['Close'], color='blue', alpha=0.7)
plt.title('Scatter Plot of High vs Close Price for TSLA (2020-2021)', fontsize=14)
plt.xlabel('High Price', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.grid(True)
plt.show()

# Presume we used yesterday's High price to predict Today's close...
# X = Yesterday's High, Y = Today's Close
X = df['High'].shift(+1)  # Yesterday's High
y = df['Close']           # Today's Close

df2 = pd.DataFrame({'High': X, 'Close': y})
df2 = df2.dropna()

X = df2['High']
y = df2['Close']

df2

# -------------------------------------------------------------------------------------
# Step 4: Train Test Split
# -------------------------------------------------------------------------------------

# Split the data (first 80% for training)
split = int(0.8 * len(X))

X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# -------------------------------------------------------------------------------------
# Step 5: Fit Linear Regression
# -------------------------------------------------------------------------------------

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Comparing predicted vs actual values
comparison = pd.DataFrame({
    'predicted': y_pred.tolist(),
    'actual': y_test.values.tolist()
})

# Display the comparison
display(comparison)

# -------------------------------------------------------------------------------------
# Step 6: R2 Score
# -------------------------------------------------------------------------------------

# Evaluate the model performance using R2 score
r2 = r2_score(y_test, y_pred)
print(f'R2 score: {r2:.4f}')

# The LR fit between Yesterday's High and Today's Close is 94%! Great fit!

# -------------------------------------------------------------------------------------
# Step 7: Prediction
# -------------------------------------------------------------------------------------

# Predict for a new "High" price
test = model.predict([[200]])

# Predicting tomorrow's close price when today's High is $200
print(f'Predicted Close price for High of $200: {test[0]:.2f}')

# -------------------------------------------------------------------------------------
# THE END
# -------------------------------------------------------------------------------------

