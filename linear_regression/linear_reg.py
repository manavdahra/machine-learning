import datetime
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import quandl
from matplotlib import style
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

style.use('ggplot')

quandl.ApiConfig.api_key = '4TEJXVxKQcCdExwC2UPy'
df = quandl.get('WIKI/GOOGL')

df['Volatility'] = (df['Adj. High'] - df['Adj. Close']) / (df['Adj. Close']) * 100
df['Change'] = (df['Adj. Close'] - df['Adj. Open']) / (df['Adj. Open']) * 100

forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forecast_out = math.ceil(0.1 * len(df))

df['label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
clf = LinearRegression(n_jobs=5)
clf.fit(x_train, y_train)

with open('../dump/trained_clf', 'wb') as f:
    pickle.dump(clf, f)

with open('../dump/trained_clf', 'rb') as F:
    clf = pickle.load(F)

accuracy = clf.score(x_test, y_test)
# print(accuracy)

forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
oneday = 86400
next_unix = last_unix + oneday

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += oneday
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
