# This notebook presents a simple way to introduce RNN (recurrent neural networks) and LSTM (long short term memory networks) for price movement predictions in trading Forex, Stock Market.
# X is downloaded from yahoo finance. Y (target for prediction) is the next day price. This algo predicts the next day price given 30 days of historical prices(backcandles = 30)
# Source: https://www.youtube.com/@CodeTradingCafe

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime


# enter your desired stock ticker, default value is 'MSFT'
ticker = 'MSFT'
# startDate , as per our convenience we can modify
startDate = datetime.datetime(2014, 12, 24)
# endDate , as per our convenience we can modify
endDate = datetime.datetime(2024, 12, 24)


# load data from yahoo finance API and transform
data = yf.download(tickers = ticker, start = startDate, end = endDate)
data['TargetNextClose'] = data['Close'].shift(-1)
data.reset_index(inplace = True)
data.drop(['Volume', 'Date'], axis=1, inplace=True)
data_set = data.iloc[:, 0:11]#.values
pd.set_option('display.max_columns', None)


# scaling the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data_set)
print(data_set_scaled)


# multiple feature from data provided to the model
X = []
backcandles = 30
print(data_set_scaled.shape[0])
for j in range(4):#data_set_scaled[0].size):#2 columns are target not X
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):#backcandles+2
        X[j].append(data_set_scaled[i-backcandles:i, j])


#move axis from 0 to position 2
X=np.moveaxis(X, [0], [2])
# Choose -1 for last column, classification else -2...
X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-1])
y=np.reshape(yi,(len(yi),1))
#y=sc.fit_transform(yi)


# split data into train test sets
splitlimit = int(len(X)*0.8)
print(splitlimit)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train)

# Build the LSTM model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
np.random.seed(10)

lstm_input = Input(shape=(backcandles, 4), name='lstm_input')
inputs = LSTM(150, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)

# Results
y_pred = model.predict(X_test)
#y_pred=np.where(y_pred > 0.43, 1,0)
for i in range(10):
    print(y_pred[i], y_test[i])

# Plot the results
plt.figure(figsize=(16,8))
plt.plot(y_test, color = 'black', label = 'true')
plt.plot(y_pred, color = 'green', label = 'pred')
plt.legend()
# adding ticker name to title
tickerName = yf.Ticker(ticker).info['shortName']
plt.title( 'True vs Predicted Stock Prices for '+ tickerName )
plt.show()


