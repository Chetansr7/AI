from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
colormap = cm.get_cmap('tab20')
cm_dark = ListedColormap(colormap.colors[::2])

# generate data
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)

points = data[0]

plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap=cm_dark)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)

kmeans.fit(points)

y_km = kmeans.fit_predict(points)

plt.figure()

plt.scatter(points[:,0], points[:,1], c=y_km, cmap=cm_dark)

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='k')

plt.show()
input()
