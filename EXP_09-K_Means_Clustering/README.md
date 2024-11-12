

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

#

```python
def kmeans(X, k, max_iters=100):
```
The function takes in three parameters:
- `X` : the dataset to be clustered
- `k` : number of clusters
- `max_iters` : maximum number of iterations (default is 100)

```python
    np.random.seed(0)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
```

`np.random.seed(0)` sets the seed for the random number generator to ensure reproducibility of the results. By setting the seed to a fixed value (in this case, 0), the same sequence of random numbers will be generated every time the code is run. This is useful for debugging and testing purposes, as it allows for consistent results.

Randomly initialize `k` centroids from the dataset `X`. The `np.random.choice` function selects `k` random indices from the range of `X.shape[0]`, and the `replace=False` argument ensures that the same index is not selected more than once.


```python
    for _ in range(max_iters):
```
The loop iterates `max_iters` times, or until convergence is reached.

```python
        clusters = [[] for _ in range(k)]
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)
```
Assign each data point to the closest cluster based on the Euclidean distance. The `euclidean_distance` function calculates the Euclidean distance between two points.

`clusters = [[] for _ in range(k)]` initializes a list of empty lists, where each list represents a cluster.

`distances = [euclidean_distance(point, centroid) for centroid in centroids]` calculates the Euclidean distance between the current point and each centroid. and iterates for every point.

Then the index of the closest centroid is found and the point is appended to the cluster.

```python
        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
```
Update the centroids of the clusters by taking the mean of all points assigned to each cluster. If the centroids have not changed, it means that the algorithm has converged and the loop can be terminated.

Else update the centroid by the new ones and continue on for next iteration

```python
    labels = np.zeros(X.shape[0])
        for cluster_index, cluster in enumerate(clusters):
            for point in cluster:
                labels[np.where((X == point).all(axis=1))] = cluster_index
```
Assign a label to each data point based on the cluster it belongs to.

```python
    return centroids, labels
```
Return the final centroids and labels of the clusters.

#### Example
```python
centroids = [[1.5, 2.5], [5.5, 6.5], [9.5, 10.5]]
labels = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
```

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
This code will output a scatter plot with the clustered data points colored according to their assigned cluster, and centroids marked with a red 'X'.

`plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')`

- `X[:, 0]` and `X[:, 1]` are the x and y coordinates of the data points, respectively.
  - The syntax `X[:, 0]` means:
    - The comma ( , ) is used to separate the row and column indices.
    - The colon ( : )  before the comma means "all rows".
    - The 0 after the comma means "the column at index 0".

- `c=labels` means that the color of each data point will be determined by its corresponding label.

- `s=50` means that the size of each data point will be 50.

- `cmap='viridis'` means that the color map used to determine the color of each data point will be the 'viridis' color map.


---
## Organizing Points into Clusters

The points are organized into clusters based on their assigned labels.

```python
clusters = [[] for _ in range(k)]
for label, point in zip(labels, X):
    clusters[int(label)].append(point)
```
This code will output a list of lists, where each sublist contains the points that belong to the same cluster.

#
#### Note

The zip function is used to iterate over two or more lists in parallel. It takes iterables (can be zero or more), aggregates them in a tuple, and returns it.

#### Example
```python
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

for i, j in zip(list1, list2):
    print(f"({i}, {j})")
```
Outputs
```python
(1,'a')
(2,'b')
(3,'c')
```

#


```python
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:")
    for point in cluster:
        print(point)
    print()
```
This code will output the points in each cluster, with the cluster number printed above each cluster.

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