#Is DBS Price Correlated to UOB Price?

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

ticker1 = input("Key in Ticker 1:")
ticker2 = input ("Key in Ticker 2:")

df1 = yf.download(ticker1, start = '2020-01-01')
df2 = yf.download(ticker2, start = '2020-01-01')

df3 = pd.concat([df1.Close, df2.Close], axis = 'columns')

print(df3)
#-----------------------------------------------------------

plt.plot(df3.index,
         df3.iloc[:,0], color = 'red')

plt.plot(df3.index,
         df3.iloc[:, 1], color = 'blue')

plt.title(f'{ticker1} is Red / {ticker2} is Blue')
plt.show()
#-----------------------------------------------------------

plt.scatter(df3.iloc[:,0],
            df3.iloc[:,1], alpha = 0.5)

plt.xlabel(ticker1, color = 'red')
plt.ylabel(ticker2, color = 'blue')
plt.show()

#-------------------------------------------------------------
from sklearn.linear_model import LinearRegression

model_1 = LinearRegression()
model_2 = LinearRegression()

df3 = df3.reset_index()
print(df3)
#-------------------------------------------

X = df3.index.values.reshape(-1,1)
y1 = df3.iloc[:, 1]
y2 = df3.iloc[:, 2]

model_1.fit(X, y1)
model_2.fit(X, y2)

predictions_1 = model_1.predict(X)
predictions_2 = model_2.predict(X)

#---------------------------------------------------
plt.scatter(df3.index,
            y1, alpha = 0.5, color = 'red')

plt.plot(df3.index,
         predictions_1, color = 'red')

plt.scatter(df3.index,
            y2, alpha = 0.5, color = 'blue')

plt.plot(df3.index,
         predictions_2, color = 'blue')


plt.title(f'{ticker1} is Red / {ticker2} is Blue')
plt.show()

#--------------------------------------------------------
model_3 = LinearRegression()
model_3.fit(y1.values.reshape(-1,1), y2)
predictions_3 = model_3.predict(y1.values.reshape(-1,1))

plt.scatter(y1, y2, alpha = 0.5)
plt.plot(y1, predictions_3, color = 'red')
plt.xlabel(ticker1)
plt.ylabel(ticker2)
plt.show()

#--------------------------------------------------------------
print(y1.corr(y2))


#THE END
