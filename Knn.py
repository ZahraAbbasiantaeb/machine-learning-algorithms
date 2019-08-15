import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math

# calculates euclideanDistance of given points
def euclideanDistance(instance1, instance2):
    distance=0
    length=len(instance2)
    for x in range(0,length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

# calculates manhatanDistance of given points
def manhatanDistance(instance1, instance2):
    distance=0
    length=len(instance2)
    for x in range(0,length):
        distance += abs(instance1[x] - instance2[x])
    return (distance)

# calculates chebyshevDistance of given points
def chebyshevDistance(instance1, instance2):
    distance=0
    length=len(instance2)
    for x in range(0,length):
        distance = max(distance, abs(instance1[x] - instance2[x]))
    return (distance)

# calculates cosineDistance of given points
def cosineDistance(instance1, instance2):
    distance=np.dot(instance1,instance2)
    distance=distance/(np.math.sqrt(np.dot(instance1, instance1))*np.math.sqrt(np.dot(instance2, instance2)))
    return (distance)

#load dataset
def loadData(path):
    data = pd.read_csv(path, sep=',',header=None, names=['f1','f2','f3','f4','type','num'])
    data=shuffle(data)

    for index,row in data.iterrows():
        if row.type=='Iris-setosa':
            data.set_value(index,'num',0)
        if row.type=='Iris-versicolor':
            data.set_value(index, 'num', 1)
        if row.type=='Iris-virginica':
            data.set_value(index, 'num', 2)

    setosa=data.loc[data.type == 'Iris-setosa']
    versicolor=data.loc[data.type == 'Iris-versicolor']
    virginica=data.loc[data.type == 'Iris-virginica']

    setosa_train, setosa_test = train_test_split(setosa, test_size=0.2)
    versicolor_train,versicolor_test=train_test_split(versicolor, test_size=0.2)
    virginica_train,virginica_test=train_test_split(virginica, test_size=0.2)

    test=setosa_test.append([versicolor_test,virginica_test])
    train=setosa_train.append([versicolor_train,virginica_train])
    return test,train,data

# get neighbours of given data point
def getNeighbors(trainData,testInstance,k,distanceType):
    distances = []
    for x in range(0,len(trainData)):

        if(distanceType=='manhatan'):
            dist = manhatanDistance(testInstance, trainData[x])
        elif(distanceType=='euclidean'):
            dist = euclideanDistance(testInstance, trainData[x])
        elif (distanceType=='chebyshev'):
            dist = chebyshevDistance(testInstance, trainData[x])
        elif (distanceType=='cosine'):
            dist = cosineDistance(testInstance, trainData[x])

        distances.append((x, dist))
    distances=sorted(distances, key=lambda t: t[1], reverse=False)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors

# calculates accuracy of prediction
def getAccuracy(testData, predictions):
    correct = 0

    for x in range(len(testData)):
        if testData[x] == predictions[x]:
            correct += 1
    return (correct / float(len(testData))) * 100.0

#normalize a dataset
def normalize(data,cols):
    for col in cols:
        maximum=max(data[:,col])
        data[:,col]/=maximum
    return data

# knn for test based on train set and it's target with k and distanceType
def Knn(test, train, target, k, distanceType):
    prediction=[]
    for row in test:
        neighbor= getNeighbors(train, row, k, distanceType)
        tmp=np.zeros(100)
        for j in range(0,k):
            tmp[int(target[neighbor[j][0]])]+=1
        prediction.append(np.argmax(tmp))
    return prediction

# split target value and data from dataset
def splitTargetAndDataset(dataset):
    length= len(dataset)
    data=np.zeros([length,4])
    target=[]
    for i in range(0,length):
        data[i, 0] = dataset[i, 0]
        data[i, 1] = dataset[i, 1]
        data[i, 2] = dataset[i, 2]
        data[i, 3] = dataset[i, 3]
        target.append(dataset[i, 5])
    return data,target

test,train,data = loadData('/iris.txt')
train_data = train.as_matrix()
test_data= test.as_matrix()
train_data,train_target=splitTargetAndDataset(train_data)
test_data,test_target=splitTargetAndDataset(test_data)
