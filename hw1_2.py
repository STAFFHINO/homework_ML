import random
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()

scaler = StandardScaler()
X_std = scaler.fit_transform(iris.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
x = X_pca

K = 3
ma = x[random.sample(range(len(x)), K)]
ro = lambda x_vect, m_vect: np.mean((x_vect - m_vect) ** 2)

COLORS = ('green', 'blue', 'brown')

plt.ion()
n = 0
last_ma = []
while not(np.array_equal(last_ma,ma)):
    X = [[] for i in range(K)]

    for x_vect in x:
        r = [ro(x_vect, m) for m in ma]
        X[np.argmin(r)].append(x_vect)
    last_ma = ma
    ma = [np.mean(xx, axis=0) for xx in X]

    plt.clf()
    # отображение найденных кластеров
    for i in range(K):
        xx = np.array(X[i]).T
        plt.scatter(xx[0], xx[1], s=10, color=COLORS[i])

    # отображение центров кластеров
    mx = [m[0] for m in ma]
    my = [m[1] for m in ma]
    plt.scatter(mx, my, s=50, color='red')

    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(1)

plt.ioff()
plt.show()
