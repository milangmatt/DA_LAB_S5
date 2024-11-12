

K-Means Clustering Implementation
==

## Introduction

This code implements the K-Means clustering algorithm to group similar data points into clusters. The algorithm iteratively updates the centroids of the clusters and assigns each data point to the closest cluster.

---
## Euclidean Distance Function

The Euclidean distance function calculates the distance between two points `a` and `b` in n-dimensional space.

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
```

Mathematically, this can be represented as:

```math
d(a, b) = √(∑(a_i - b_i)^2)
```

where `a_i` and `b_i` are the i-th components of the vectors `a` and `b`, respectively.

---
## K-Means Algorithm Implementation

The K-Means algorithm implementation takes in the dataset `X`, the number of clusters `k`, and the maximum number of iterations `max_iters`.

```python
def kmeans(X, k, max_iters=100):
    # Randomly initialize centroids
    np.random.seed(0)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign clusters
        clusters = [[] for _ in range(k)]
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)
        
        # Update centroids
        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    # Assign labels
    labels = np.zeros(X.shape[0])
    for cluster_index, cluster in enumerate(clusters):
        for point in cluster:
            labels[np.where((X == point).all(axis=1))] = cluster_index
    
    return centroids, labels
```

The algorithm iteratively updates the centroids of the clusters using the following steps:

1. Assign each data point to the closest cluster based on the Euclidean distance.
2. Update the centroids of the clusters by taking the mean of all points assigned to each cluster.
3. Check for convergence by comparing the old and new centroids. If they are the same, the algorithm terminates.

---
## Loading the Dataset

The dataset is loaded from a CSV file using the `pd.read_csv` function.

```python
data = pd.read_csv('data.csv')
X = data.values
```

---
## Running K-Means

The K-Means algorithm is run with the loaded dataset and the specified number of clusters.

```python
centroids, labels = kmeans(X, k)
```

---
## Visualizing the Clustered Data

The clustered data is visualized using a scatter plot, where each point is colored according to its assigned cluster.

```python
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering")
plt.xlabel("VAL1")
plt.ylabel("VAL2")
plt.show()
```

---
## Organizing Points into Clusters

The points are organized into clusters based on their assigned labels.

```python
clusters = [[] for _ in range(k)]
for label, point in zip(labels, X):
    clusters[int(label)].append(point)
```

The clusters are then printed to the console.

```python
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:")
    for point in cluster:
        print(point)
    print()
```

---

Complete Code
--
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to calculate the Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# K-Means algorithm implementation
def kmeans(X, k, max_iters=100):
    # Randomly initialize centroids
    np.random.seed(0)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign clusters
        clusters = [[] for _ in range(k)]
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in  centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)
        
        # Update centroids
        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    # Assign labels
    labels = np.zeros(X.shape[0])
    for cluster_index, cluster in enumerate(clusters):
        for point in cluster:
            labels[np.where((X == point).all(axis=1))] = cluster_index
    
    return centroids, labels

# Load the dataset from the CSV file
data = pd.read_csv('data.csv')
X = data.values
# Set the number of clusters
k = 3

# Run K-Means
centroids, labels = kmeans(X, k)

# Visualize the clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering")
plt.xlabel("VAL1")
plt.ylabel("VAL2")
plt.show()

# Organize points into clusters
clusters = [[] for _ in range(k)]
for label, point in zip(labels, X):
    clusters[int(label)].append(point)

# Display the clusters as lists
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:")
    for point in cluster:
        print(point)
    print()

```