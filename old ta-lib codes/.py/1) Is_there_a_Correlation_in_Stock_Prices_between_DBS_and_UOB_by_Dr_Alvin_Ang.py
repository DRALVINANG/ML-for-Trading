#--------------------------------------------------------------------------------------
# Step 1: Import Libraries
#--------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from sklearn.linear_model import LinearRegression

#--------------------------------------------------------------------------------------
# Step 2: Import DBS and UOB Stock Prices
#--------------------------------------------------------------------------------------

# 2a) Setting the Start and End Dates
start_date = '2020-01-01'

# Downloading stock data
data1 = yf.download('D05.SI', start=start_date)  # DBS
data2 = yf.download('U11.SI', start=start_date)  # UOB

data1.columns = data1.columns.droplevel(level=1)
data2.columns = data2.columns.droplevel(level=1)

# Preview the data
print("DBS Data Preview:")
print(data1.head())
print("\nUOB Data Preview:")
print(data2.head())

#--------------------------------------------------------------------------------------
# Step 3: Placing Both Stocks into ONE Dataframe
#--------------------------------------------------------------------------------------

data = pd.merge(data1.Close, data2.Close, left_index=True, right_index=True)
data.rename(columns={'Close_x': 'DBS Close Price', 'Close_y': 'UOB Close Price'}, inplace=True)
print("\nMerged Dataframe:")
print(data.head())

#--------------------------------------------------------------------------------------
# Step 4: Plot
#--------------------------------------------------------------------------------------

plt.figure(figsize=(12, 5))
plt.scatter(data['DBS Close Price'], data['UOB Close Price'])
plt.xlabel('DBS price')
plt.ylabel('UOB price')
plt.title('Scatter Plot of DBS vs UOB Stock Prices')
plt.show()

#--------------------------------------------------------------------------------------
# Step 5: Formulating the Linear Regression Model
#--------------------------------------------------------------------------------------

# 5a) Assign UOB into y and DBS into X
y = data['UOB Close Price'].values.reshape(-1, 1)
X = data['DBS Close Price'].values.reshape(-1, 1)

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients
print(f"Intercept: {model.intercept_[0]}")
print(f"Coefficient: {model.coef_[0][0]}")

# Print the R^2 score
print(f"R^2 score: {model.score(X, y)}")

# Predict a value
predicted_value = model.predict([[38]])
print(f"Predicted UOB Close Price for DBS price of 38: {predicted_value[0][0]}")

#--------------------------------------------------------------------------------------
# Step 6: Final Plot of the Linear Regression Equation
#--------------------------------------------------------------------------------------

plt.figure(figsize=(12, 5))
plt.scatter(data['DBS Close Price'], data['UOB Close Price'], label='Data Points')
plt.plot(data['DBS Close Price'], model.intercept_[0] + model.coef_[0][0] * data['DBS Close Price'], color='red', label='Regression Line')
plt.xlabel('DBS price')
plt.ylabel('UOB price')
plt.title('Linear Regression: DBS vs UOB Stock Prices')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------
# THE END
#--------------------------------------------------------------------------------------

