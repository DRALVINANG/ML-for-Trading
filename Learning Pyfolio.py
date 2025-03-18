#--------------------------
# Step 1: Install and Import Libraries
#--------------------------

#THONNY MUST HAVE THESE VERSIONS INSTALLED OR PYFOLIO WILL NOT WORK!!!
#import os
#os.system("pip install numpy==1.23.0")
#os.system("pip install pandas==1.3.5")


# Import libraries
import yfinance as yf
import pandas as pd
import pyfolio as pf
import matplotlib.pyplot as plt

# Set matplotlib style
plt.style.use('fivethirtyeight')

#--------------------------
# Step 2: Download DBS Data from Yahoo Finance
#--------------------------

# Define the start date
start_date = '2023-01-01'

# Download DBS historical data (Singapore stock)
DBS_history = yf.download('D05.SI', start=start_date)

# Display the 'Close' price of DBS
DBS_history['Close']

#--------------------------
# Step 3: Plot DBS History
#--------------------------

# Plot the 'Close' price of DBS
DBS_history['Close'].plot(figsize=(20, 5))
plt.title('DBS Stock Price')
plt.ylabel('Price (SGD)')
plt.xlabel('Date')
plt.show()

#--------------------------
# Step 4: Obtain DBS Daily Returns
#--------------------------

# Calculate daily returns from the 'Close' price
DBS_returns = DBS_history.Close.pct_change()

# Display the calculated returns
print(DBS_returns)

# Plot daily returns as a bar plot
DBS_returns.plot(kind='bar', figsize=(20, 5), title="DBS Daily Returns")
plt.show()

# Flatten DBS_returns if it's a 2D array (just in case)
DBS_returns = pd.Series(DBS_returns.squeeze())

#--------------------------
# Step 5: Use Pyfolio to Create Simple Tear Sheet
#--------------------------
pf.timeseries.cum_returns(DBS_returns).plot()
plt.ylabel('DBS Returns')
plt.show()

perf_stats = pf.timeseries.perf_stats(DBS_returns)
print(perf_stats)

# Generate a simple Pyfolio tear sheet for the DBS returns
pf.create_simple_tear_sheet(DBS_returns)
plt.show()

#--------------------------
# End of Script
#--------------------------
