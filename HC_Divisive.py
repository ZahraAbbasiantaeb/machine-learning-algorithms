from random import randint
import numpy as np
from q1.Kmeans import Kmean
from util import  plot_clusterede_data_2D


def select_centroid(cluster_count, points):

    centroids = []

    for i in range(0, cluster_count):
        index = randint(0, len(points) - 1)
        centroids.append(points[index])

    return centroids


def partition_cluster(points, clusters, cluster_count):

    parts = []

    for i in range(0, cluster_count):

        cluster = np.zeros((clusters.count(i), 2))
        index = 0

        for j in range(0, len(points)):

            if clusters[j] == i:
                cluster[index] = points[j]
                index += 1

        parts.append(cluster)

    return parts


def HC_Divisive(points, level):

    cluster_count = 2
    clustered_points = []
    clustered_points.append(points)
    level_cluster = []

    for i in range(0, level):

        final_clusters = []

        for cluster in clustered_points:

            clusters, _, _, _, _ = Kmean(cluster, select_centroid(cluster_count, cluster))
            new_clusters = partition_cluster(cluster, clusters, cluster_count)

            for t1 in new_clusters:
                final_clusters.append(t1)

        clustered_points = final_clusters
        level_cluster.append(final_clusters)

    return final_clusters,level_cluster


# test this algorithm

points = np.loadtxt('/Users/zahra_abasiyan/Desktop/Machine Learning/HW5/question3/data_h.txt',delimiter='\t' )

level=3

clustered_data,level_cluster = HC_Divisive(points, level)

for cluster in level_cluster:
    plot_clusterede_data_2D(cluster)
