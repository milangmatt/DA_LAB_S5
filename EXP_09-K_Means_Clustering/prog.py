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
