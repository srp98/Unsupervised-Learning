import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# Create a random seed and set it to 0
np.random.seed(0)

# Make random clusters using blobs
X, y = make_blobs(n_samples=5000, centers=[[2, 1], [-4, -2], [1, -4], [0, 3]], cluster_std=0.7)

# Scatter plot the data to check
plt.scatter(X[:, 0], X[:, 1], marker='.', edgecolors='white')
plt.show()

""" 
Set K-Means Clustering, 
n_clusters is number of clusters to form, 
k-means++ for smart way to converge faster
n_init initializes different centroids each run and best output is shown in terms of inertia
"""

k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)

# Fit KMeans model with feature matrix, X
k_means.fit(X)

# Grab the labels for each point in the model
k_means_labels = k_means.labels_
print(k_means_labels)

# Get the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)

# Visualize the Plot
# Initialize plot with the specified dimensions.
fig = plt.figure(figsize=(8, 5))

# Colors uses a color map, which will produce an array of colors based on the number of labels there are.
colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot with a black background for better visibility
ax = fig.add_subplot(1, 1, 1, facecolor='black')

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each data point is in.
for k, col in zip(range(len([[2, 1], [-4, -2], [1, -4], [0, 3]])), colors):
    # Create a list of all data points, where the data points that are in the cluster are labeled as true
    # Else they are labeled as false.
    members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the data points with color col.
    ax.plot(X[members, 0],
            X[members, 1],
            'w',
            markerfacecolor=col,
            marker='.')

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0],
            cluster_center[1],
            'o',
            markerfacecolor=col,
            markeredgecolor='k',
            markersize=6)

# Title of the plot
ax.set_title('KMeans-Clustering')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

# Display the scatter plot from above for comparison.
plt.scatter(X[:, 0], X[:, 1], marker='+')
