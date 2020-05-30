import pandas as pd 
import quandl, math, datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style

style.use('ggplot')
shiftlen = 0.005
outlierlabel = -99999
tsize = 0.2
sec_per_day = 86400
tick = 'GOOGL'

df = quandl.get(('WIKI/' + tick))
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])*100/ df['Adj. Low']
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])*100/ df['Adj. Open']
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast = 'Adj. Close'
df.fillna(outlierlabel, inplace=True)
forecast_out = int(math.ceil(shiftlen*len(df)))

df['label'] = df[forecast].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_recent = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)


forecast_set = clf.predict(X_recent)

print(forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
next_unix = last_unix + sec_per_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += sec_per_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()