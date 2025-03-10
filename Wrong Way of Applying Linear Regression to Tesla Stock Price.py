# -------------------------------------------------------------------------------------
# Step 1: Install Libraries
# -------------------------------------------------------------------------------------
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------------------------
# Step 2: Import Data from 2020 to 2021
# -------------------------------------------------------------------------------------

ticker = 'TSLA'
df = yf.download(ticker, start='2020-01-01', end='2021-12-31')

# Dropping the multi-level column index after downloading the data
df.columns = df.columns.droplevel(level=1)

# Reset the index and rename it to 'Day'
df = df.reset_index()
df = df.reset_index()
df = df.rename(columns={'index': 'Day'})

# Display the DataFrame
df

# -------------------------------------------------------------------------------------
# Step 3: Create New DataFrame with Day and Close Price
# -------------------------------------------------------------------------------------

data = {'Day': df.Day, 'Close': df.Close}
new_df = pd.DataFrame(data)

# Display the new DataFrame
new_df

# -------------------------------------------------------------------------------------
# Step 4: Plotting the Bull Run from 2020 to 2021
# -------------------------------------------------------------------------------------

sns.lmplot(x='Day', y='Close', data=new_df)

# Display the plot
plt.show()  # This ensures that the plot is displayed in Thonny or other IDEs

# -------------------------------------------------------------------------------------
# Step 5: Linear Regression Prediction for Day 666 on TSLA
# -------------------------------------------------------------------------------------

# Initialize and fit the linear regression model
lm = LinearRegression()

# Reshape X and y for the Linear Regression Model
X = new_df.Day.values.reshape(-1, 1)
y = new_df.Close.values.reshape(-1, 1)

# Fit the model
lm.fit(X, y)

# Print coefficients and intercept
print("Coefficient:", lm.coef_)
print("Intercept:", lm.intercept_)

# The Linear Equation: Close Price = 0.63(Day) + 20.2

# Try predicting the price on day 666
a = [[666]]
Predicted_Price_Day666 = lm.predict(a)
print("Predicted price for Day 666:", Predicted_Price_Day666)

# -------------------------------------------------------------------------------------
# Step 6: Check TSLA Performance in 2022
# -------------------------------------------------------------------------------------

stock2022 = yf.download(ticker, start='2022-01-01', end='2022-12-31')

# Reset index and display the stock data for 2022
stock2022 = stock2022.reset_index()
stock2022 = stock2022.reset_index()

# Display the stock data for 2022
stock2022

# Plotting the stock performance for 2022
sns.lmplot(x='index', y='Close', data=stock2022)

# Display the plot
plt.show()  # This ensures that the plot is displayed in Thonny or other IDEs

# Check the actual price on the corresponding day (Day 163)
actual_price_2022 = stock2022.iloc[163].Close
print("Actual price on Day 163 in 2022:", actual_price_2022)

# -------------------------------------------------------------------------------------
# Step 7: Conclusion
# -------------------------------------------------------------------------------------

# Conclusion:
# - We cannot blindly use Linear Regression to predict stock prices.
# - Time Series Data has Autocorrelation, which makes Linear Regression unsuitable for forecasting.
# - Autocorrelation means that the variable is related back to itself, which violates the assumption of Linear Regression.

print("Conclusion: Time Series data requires more sophisticated models for forecasting.")

