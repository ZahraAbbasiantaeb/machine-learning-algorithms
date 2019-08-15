import numpy as np
from Kmeans import getEuclidDistance
from util import plot_clusters_2D

points = np.loadtxt('/data_h.txt',delimiter='\t' )


def find_cluster_members(cluster, points, lables):

    length = len(points)
    members = []

    for i in range(0, length):
        if lables[i] == cluster:
            members.append(points[i])

    return members


def find_distances(cluster1, cluster2, dist_measure):

    dist=[]

    for point1 in cluster1:
        for point2 in cluster2:
            dist.append(getEuclidDistance(point1, point2))

    if (dist_measure == 'single'):

        return np.min(dist)

    elif (dist_measure == 'average'):

        return np.average(dist)

    elif (dist_measure == 'complete'):

        return np.max(dist)

    return np.mean(dist)


def update_lable(lables, cluster1_ID, cluster2_ID):

    length= len(lables)

    for i in range (0, length):
        if lables[i]==cluster2_ID:
            lables[i]=cluster1_ID

    return lables


def HC_Agglomerative(points, cluster_size, dist_type):

    lables = list(range(len(points)))
    current_cluster_size = len(points)

    while (current_cluster_size > cluster_size):

        clusters = list(set(lables))
        length = len(clusters)
        dist = np.math.inf
        cluster1_num = 0
        cluster2_num = 0

        for i in range(0, length):
            for j in range(i + 1, length):

                cluster1 = find_cluster_members(clusters[i], points, lables)
                cluster2 = find_cluster_members(clusters[j], points, lables)
                distance = find_distances(cluster1, cluster2, dist_type)

                if (dist > distance):
                    dist = distance
                    cluster1_num = clusters[i]
                    cluster2_num = clusters[j]

        # print(cluster1_num)
        # print(cluster2_num)
        # print('**********')
        lables = update_lable(lables, cluster1_num, cluster2_num)
        current_cluster_size=len(set(lables))
        print(current_cluster_size)
    return lables


labels = HC_Agglomerative(points, 8, 'single')


plot_clusters_2D(labels, points)

