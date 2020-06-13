# Mega Case Study - Make a Hybrid Deep Learning Model

# Part 1 - Identify the frauds with SOM
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler, StandardScaler

dataset = pd.read_csv("Credit_Card_Applications.csv")

print(dataset.info())
print(dataset.describe())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

print("Som Trained")

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)
frauds = sc.inverse_transform(frauds)

# Part 2 - Going from unsupervised to Supervised deep learning
customers = dataset.iloc[:, 1:].values

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

sc = StandardScaler()
customers = sc.fit_transform(customers)

classifier = Sequential()

classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(customers, is_fraud, batch_size=10, epochs=100)

y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)

y_pred = y_pred[y_pred[:, 1]].argsort()
