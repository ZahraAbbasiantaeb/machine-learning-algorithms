import numpy as np

cluster_count = 2


# measures Euclidean distance of two points
def getEuclidDistance(p,center):

    dist=0

    for i in range (len(center)):
        dist+=(p[i]-center[i])**2

    return dist


# find nearest cluster to given point
def get_closest_cluster(p, centers):

    best = 0
    closest = np.inf

    for idx in range(len(centers)):
        temp_dist = getEuclidDistance(p,centers[idx])

        if temp_dist < closest:
            closest = temp_dist
            best=idx

    return best, closest


# moves the centroids of cluster to mean of the points in that cluster
def updateCentroid(points, clusters):

    centroids=[]
    for i in range(0, cluster_count):

        individuals = np.zeros((clusters.count(i),2))
        index = 0

        for j in range(0, len(points)):
            if clusters[j] == i:
                individuals[index] = points[j]
                index+= 1

        centroids.append(get_centroid(individuals))

    return centroids


# finds the centroid of the given points
def get_centroid(individuals):

    length= len(individuals[0])
    center=np.zeros(length)

    for j in range(len(individuals)):
        for i in range (0, length):
            center[i]+=individuals[j][i]

    for i in range(0, length):
        center[i]/=len(individuals)

    return center


# assigns each point to corresponding cluster
def assign_clusters(points, centers):

    clusters = []
    distance = []

    if (len(points) < len(centers)):
        return None

    for point in points:
        tmp=get_closest_cluster(point, centers)
        clusters.append(tmp[0])
        distance.append(tmp[1])

    return clusters, distance


# calculates distance of prev and current centroids
def get_centroids_dist(prev_centroid, curr_centroids):

    dist=0

    for index in range (0, len(curr_centroids)):
        dist+= getEuclidDistance(prev_centroid[index], curr_centroids[index])

    return dist


# returns each clusters points, based on given lables
def get_each_clusters_points(labels, points):

    arr=[]

    for i in range(0, cluster_count):

        individuals = np.zeros((labels.count(i), 2))
        index = 0

        for j in range(0, len(points)):
            if labels[j] == i:
                individuals[index] = points[j]
                index+= 1
        arr.append(individuals)

    return arr


# returns DBI indicator of clusters
def DaviesBouldin(clusters, distances, centroids):

    variances=[]
    db=[]
    for i in range(0, cluster_count):

        individuals = []

        for j in range(0, len(clusters)):
            if clusters[j] == i:
                individuals.append(distances[j])

        variances.append(np.mean(individuals))

    for i in range(0, cluster_count):
        for j in range(0, cluster_count):
            if(j != i):
                db.append((variances[i] + variances[j]) / getEuclidDistance(centroids[i], centroids[j]))

    return np.mean(db)


def Kmean(points, centroids):

    SSE=[]
    DBI=[]
    centroids_per_ittr=[]

    while (True):

        centroids_per_ittr.append(centroids)
        prev_centroids = centroids
        clusters, distances = assign_clusters(points, centroids)
        centroids = updateCentroid(points, clusters)
        SSE.append(np.sum(distances))
        db = DaviesBouldin(clusters, distances, centroids)
        DBI.append(db)

        if (get_centroids_dist(prev_centroids, centroids) == 0):
            break

    return clusters, centroids, DBI, SSE, centroids_per_ittr