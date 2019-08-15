import numpy as np


# DBSCAN algorithm implementation
def DBscan(dataPoints, eps, MinPts):

    labels = np.zeros(len(dataPoints))
    Cluster_number = 0

    for curr_point in range(0, len(dataPoints)):

        if not (labels[curr_point] == 0):
            continue

        NeighborPts = find_neighbours(dataPoints, curr_point, eps)

        if len(NeighborPts) < MinPts:
            labels[curr_point] = -1

        else:
            Cluster_number += 1
            expand_cluster(dataPoints, labels, curr_point, NeighborPts, Cluster_number, eps, MinPts)

    return labels


# expands a cluster of given point
def expand_cluster(D, labels, P, NeighborPts, C, eps, MinPts):

    labels[P] = C
    i = 0
    while i < len(NeighborPts):

        Pn = NeighborPts[i]

        if labels[Pn] == -1:
            labels[Pn] = C

        elif labels[Pn] == 0:
            labels[Pn] = C

            PnNeighborPts = find_neighbours(D, Pn, eps)

            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts

        i += 1
    return


# find neighbours of given point P between Dataset D within eps distance
def find_neighbours(D, P, eps):

    neighbors = []

    for Pn in range(0, len(D)):

        if np.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)

    return neighbors


# calculates purity indicator of given clusters
def get_purity(lables, actual_lables):

    purity = 0
    length = len(lables)
    used = []

    for act_lable in set(actual_lables):

        common = 0

        for lable in set(lables):
           tmp = 0
           index = -2

           if not (lable in used):
               for i in range (0,length):

                   if lables[i] == lable and actual_lables[i] == act_lable and lables[i] != -1:
                       tmp += 1

               if (tmp >common):
                    common = tmp
                    index = lable

        purity += common
        used.append(lable)

    return (purity/length)

