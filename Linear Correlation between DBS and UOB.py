# -------------------------------------------------------------------------------------
# Step 1: Install Libraries
# -------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from sklearn.linear_model import LinearRegression

# Set the plot style for better visualization
plt.style.use('ggplot')

# To ignore warnings
import warnings
warnings.simplefilter('ignore')

# -------------------------------------------------------------------------------------
# Step 2: Import DBS and UOB Stock Prices
# -------------------------------------------------------------------------------------

# Set start and end dates for stock data
start_date = '2020-01-01'

# Download stock data for DBS (D05.SI) and UOB (U11.SI)
data1 = yf.download('D05.SI', start=start_date)
data2 = yf.download('U11.SI', start=start_date)

# Handle the multi-level columns returned by yfinance and drop the unnecessary level
data1.columns = data1.columns.droplevel(level=1)
data2.columns = data2.columns.droplevel(level=1)

# -------------------------------------------------------------------------------------
# Step 3: Preview the Data
# -------------------------------------------------------------------------------------

# Previewing DBS and UOB stock data
print("Preview of DBS data:")
print(data1.tail())

print("Preview of UOB data:")
print(data2.tail())

# -------------------------------------------------------------------------------------
# Step 4: Prepare the Data
# -------------------------------------------------------------------------------------

# Merge both stocks on the date index and select the 'Close' prices
data = pd.merge(data1['Close'], data2['Close'], left_index=True, right_index=True)

# Rename the columns for clarity
data.rename(columns={'Close_x': 'DBS Close Price', 'Close_y': 'UOB Close Price'}, inplace=True)

# -------------------------------------------------------------------------------------
# Step 5: Scatter Plot of Stock Prices
# -------------------------------------------------------------------------------------

# Plotting the scatter plot to visualize the relationship between the two stock prices
plt.figure(figsize=(12, 5))
plt.scatter(data['DBS Close Price'], data['UOB Close Price'])
plt.xlabel('DBS Close Price')
plt.ylabel('UOB Close Price')
plt.title('Scatter Plot: DBS vs UOB Stock Prices')
plt.show()

# -------------------------------------------------------------------------------------
# Step 6: Linear Regression Model using scikit-learn
# -------------------------------------------------------------------------------------

# Assign UOB prices as the dependent variable (y) and DBS prices as the independent variable (X)
X = data[['DBS Close Price']]  # Independent variable (DBS)
y = data['UOB Close Price']   # Dependent variable (UOB)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Get the model's coefficients and intercept
intercept = model.intercept_
slope = model.coef_[0]

# Display the model summary (equation)
print(f"Regression Equation: y = {intercept:.2f} + {slope:.2f} * X")
print(f"R-squared: {model.score(X, y):.4f}")

# -------------------------------------------------------------------------------------
# Step 7: Plot the Regression Line
# -------------------------------------------------------------------------------------

# Plotting the regression line on top of the scatter plot
plt.figure(figsize=(12, 5))
plt.scatter(data['DBS Close Price'], data['UOB Close Price'])
plt.plot(data['DBS Close Price'], intercept + slope * data['DBS Close Price'], color='red', label='Regression Line')
plt.xlabel('DBS Close Price')
plt.ylabel('UOB Close Price')
plt.title('Linear Regression: DBS vs UOB Stock Prices')
plt.legend()
plt.show()

# -------------------------------------------------------------------------------------
# End of the Process
# -------------------------------------------------------------------------------------

