import math

#impelements knn regression
def Knn_regression(test, train, target, k):
    prediction=[]
    for row in test:
        neighbor = getNeighbors_KnnRegression(train, row, k)
        tmp=0
        for j in range(0, k):
            tmp+=target[neighbor[j][0]]
            tmp /= k
        prediction.append(tmp)
    return prediction

# finds nearest neighbours
def getNeighbors_KnnRegression(trainData, testInstance, k):
    distances = []
    for x in range(0,len(trainData)):
        dist = euclideanDistance_knnRegression(testInstance, trainData[x])
        distances.append((x, dist))
    distances=sorted(distances, key=lambda t: t[1], reverse=False)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors

def euclideanDistance_knnRegression(instance1, instance2):
    distance = pow((instance1 - instance2), 2)
    return math.sqrt(distance)

def MSE(instance1, instance2):
    tmp=0
    index=len(instance1)
    for i in range (0,index):
        tmp+=math.pow((instance1[i]-instance2[i]),2)
    return math.sqrt(tmp)
