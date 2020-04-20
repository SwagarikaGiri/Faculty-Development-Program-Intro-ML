from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

iris_dataset = load_iris()
print(iris_dataset)
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))

df=iris_dataset['data']
sepal_length=df[0:,0]
sepal_width=df[0:,1]
petal_length=df[0:,2]
petal_width=df[0:,3]
plt.plot()
plt.title('Dataset')
plt.scatter(sepal_length,petal_width)
plt.show()

X = np.array(list(zip(sepal_length, petal_width)))
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']
plt.ylabel('Length')


kmeans = KMeans(n_clusters=3).fit(X)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')


for i, l in enumerate(kmeans.labels_):
    plt.plot(sepal_length[i], petal_width[i], color=colors[l], marker=markers[l])
plt.xlabel('Width')
plt.legend()
plt.show()