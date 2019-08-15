from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# returns each clusters points
def get_each_clusters_points(clusters, points):

    arr=[]
    cluster_count=set(clusters)

    for i in cluster_count:

        individuals = points[clusters==i]
        arr.append(individuals)

    return arr


# plots figure of clusters in 2D
def plot_clusters_2D(labels, points):

    clusters_count = len(set(labels))

    clustered_points = get_each_clusters_points(labels, points)

    color_index = 0

    colors = []

    for i in range (0, clusters_count):
        colors.append('%06X' % randint(0, 0xFFFFFF))

    for elm in clustered_points:
        plt.scatter(elm[:, 0], elm[:, 1], c='#' + colors[color_index])
        color_index += 1

    plt.show()

    return


# plots figure of clusters in 2D
def plot_clusterede_data_2D(clustered_points):

    clusters_count = len( clustered_points)

    color_index = 0

    colors = []

    for i in range (0, clusters_count):
        colors.append('%06X' % randint(0, 0xFFFFFF))

    for elm in clustered_points:
        plt.scatter(elm[:, 0], elm[:, 1], c='#' + colors[color_index])
        color_index += 1

    plt.show()

    return


# plots figure of clusters in 2D and indicates centroids
def plot_cluster_2D_with_centers(labels, points, centroids):

    clusters_count = len(set(labels))

    colors=[]

    color_index = 0

    for i in range(clusters_count + 2):
        colors.append('%06X' % randint(0, 0xFFFFFF))


    clustered_data = get_each_clusters_points(labels, points)

    for elm in clustered_data:
        plt.scatter(elm[:, 0], elm[:, 1], c='#' + colors[color_index])
        color_index += 1

    for row in centroids:
        plt.scatter(row[0], row[1], c='#' + colors[color_index], marker="*")

    plt.show()

    return


# plots figure of clusters in 3D
def plot_cluster_3D(labels, points):

    clustered_points = get_each_clusters_points(labels, points)

    colors=[]

    index = 0

    for i in range(len(clustered_points)):
        colors.append('%06X' % randint(0, 0xFFFFFF))

    fig = plt.figure()

    ax = Axes3D(fig)

    for elm in clustered_points:
        ax.scatter(elm[:, 0], elm[:, 1], elm[:, 2], c='#' + colors[index])
        index += 1

    plt.show()

    return