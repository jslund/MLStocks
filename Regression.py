import pandas as pd
import quandl, math
import os
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression

#ENV



df = quandl.get("WIKI/GOOGL",)

#print(df.head())

df = df[['Adj. Open' , 'Adj. High' , 'Adj. Low' , 'Adj. Close' , 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) # creates a default value for any NaN values
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out) # not sure about this, uses numpy to create a new column


df.dropna(inplace=True)#removes any NaN data (blanks?)

X = np.array(df.drop(['label'], 1)) # defines the numpy array as everything in df except the 'label' column - this is called a 'feature';
y = np.array(df['label']) # funnily enough this is the label column



X = preprocessing.scale(X)

X_lately = X[-forecast_out:]

X = X[:-forecast_out]





X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)


clf = LinearRegression(n_jobs=-1) #Run again with Linear Regession

clf.fit(X_train,y_train)

confidence = clf.score(X_test,y_test)

print(confidence)

forecast_set = clf.predict(X_lately)

print(forecast_set,confidence,forecast_out)
