import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

n_layers = 3
layer_units = 50
dropout_rate = 0.2

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train[["Open"]].values

sc = MinMaxScaler(feature_range=(0, 1), copy=True)
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# reshape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()

regressor.add(LSTM(units=layer_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=dropout_rate))

for i in range(n_layers):
    return_sequences = i < (n_layers - 1)
    regressor.add(LSTM(units=layer_units, return_sequences=return_sequences))
    regressor.add(Dropout(rate=dropout_rate))

regressor.add(Dense(units=1, kernel_initializer='uniform', activation="sigmoid"))

regressor.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# ###################################3
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test[["Open"]].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[(len(dataset_total) - len(dataset_test)) - 60:].values
inputs = inputs.reshape(-1, 1)

inputs = sc.transform(inputs)

X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = sc.inverse_transform(regressor.predict(X_test))

plt.plot(real_stock_price, color="red", label='Real Google Stock price')
plt.plot(predicted_stock_price, color="blue", label='Predicted Google Stock price')
plt.title("Google Stock price prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()