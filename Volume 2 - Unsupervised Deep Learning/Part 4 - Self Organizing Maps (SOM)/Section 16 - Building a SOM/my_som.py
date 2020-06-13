
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import bone, pcolor, colorbar, plot, show

from minisom import MiniSom

dataset = pd.read_csv("Credit_Card_Applications.csv")

print(dataset.info())
print(dataset.describe())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

sc = MinMaxScaler(feature_range=(0, 1),copy=True)
X = sc.fit_transform(X)


som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data=X)
som.train_random(data=X, num_iteration=100)
print("Som Trained")

bone()
pcolor(som.distance_map().T)
colorbar()

markers = ["o", "s"]
colors = ["r", "g"]

for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=30,
         markeredgewidth=2)

show()

mappings = som.win_map(data=X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)