# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:24:46 2022

@author: ASUS
"""

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:,:-1].values

X.shape

from sklearn.cluster import KMeans
ilist = []
n = 5

for i in range(1,n):
     kmeans = KMeans(n_clusters = i)
     kmeans.fit(X)
     ilist.append(kmeans.inertia_)
     
     
plt.plot(range(1,n), ilist)
plt.title('Elbow')
plt.xlabel('clusters')
plt.ylabel('inertias')
plt.show()


kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10, c = 'r')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10, c = 'b')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 10, c = 'g')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'y')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()