import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans

style.use('ggplot')

x = np.array(
    [
        [1, 2],
        [1.5, 1.8],
        [5, 8],
        [8, 8],
        [1, 0.6],
        [9, 11],
    ]
)

plt.scatter(x[:, 0], x[:, 1], s=150)

clf = KMeans(n_clusters=2)
clf.fit(x)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['g.', 'r.', 'c.', 'b.', 'k.', 'g.', 'r.', 'c.', 'b.', 'k.']

for i in range(len(x)):
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize=25)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150)
plt.show()
