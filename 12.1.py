# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:56:26 2018

@author: Harekrishna
"""
# Import Liabraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import pandas as pd
# Loading IRIS dataset
from sklearn.datasets import load_iris
iris = load_iris()
df=pd.DataFrame(iris['data'],columns=iris['feature_names'])
# Scaling of Input data using StandardScalar
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)
# Dimensionality reduction from 4D data to3D data
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
scaled_data.shape
x_pca.shape

# Visualisization
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig)

# Reorder the labels to have colors matching the cluster results
ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=iris['target'] , edgecolor='k')
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()