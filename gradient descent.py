import numpy as np
from numpy import empty
from numpy import loadtxt
from numpy.linalg import inv

# this function is implementation of gradient descent without regularization
# it returns weights and MSE in each iteration.
def gradientDescent(x,y,theta,alpha,m,numIterations):

    MSE = empty([numIterations,1])
    thetas= empty([numIterations,np.shape(x)[1]])
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (m)
        MSE[i,0]=cost
        # avg gradient per example
        gradient = np.dot(loss.transpose(),x) / m
        # update
        theta = theta - alpha * (gradient.transpose())
        thetas[i, :] = theta.transpose()
    return theta, MSE, thetas

# this function calculates MSE for given data set according to the value of the weights
# in each iteration
def testValidationSet(x_validation, y_vallidation,thetas,numIterations):

    size_validation=np.shape(x_validation)[0]
    MSEofValidation = empty([numIterations, 1])
    for i in range(0, numIterations):
        theta = empty([np.shape(thetas)[1], 1])
        for j in range(0 , np.shape(thetas)[1] ):
            theta[j,0]=thetas[i,j]
        hypothesis=np.dot(x_validation, theta)
        loss = hypothesis - y_vallidation
        cost = np.sum(loss ** 2) / (size_validation)
        MSEofValidation[i, 0] = cost
    return MSEofValidation


# this function calculates weights and MSE per each iteration
# this function is implementation of gradient descent with regularization
def gradientDescentWithRegularization(x,y,theta,alpha,m,numIterations,landa):
    MSE = empty([numIterations,1])
    thetas= empty([numIterations,np.shape(x)[1]])
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (m)
        MSE[i,0]=cost
        gradient = np.dot(loss.transpose(),x) / m
        theta = theta*(1-(alpha*landa)/m ) - alpha * (gradient.transpose())
        thetas[i, :] = theta.transpose()
    return theta, MSE, thetas

# this function initialize the input of X values given degree of function
# for example if your model is with 4 degree, it calculates and returns a matrix
# with four columns like this: X,X^2,X^3,X^4
def createNDimentionalArray(dt, n):
    x= empty([(dt.shape[0]),n])
    y=empty([(dt.shape[0]),1])
    for i in range (n):
        x[:,i]=dt[:,0]**(i+1)
    y[:,0]=dt[:,1]
    return x,y

# this function calculates the weights with OLS(Ordinary least square) method
def OrdinaryleastSquares(x,y):
    newX=np.dot(x.transpose(),x)
    invX=inv(newX)
    a=np.dot(invX,x.transpose())
    b=np.dot(a,y)
    return b

# this function initialize the weights to one
def initTheta(k):
    theta = empty([k, 1])
    theta[:, 0] = 1
    return  theta

# this function reads data from file and after shuffling the data set, patitions the
# data set to test, train, and validation data
def initDataSet():
    lines = loadtxt("/data1.txt", comments="#",
                    delimiter=",", unpack=False)
    np.random.shuffle(lines)
    train, validate, test = np.split(lines, [int(.6 * len(lines)), int(.8 * len(lines))])
    return  train,validate,test,lines

# this function calculates MSE for one set of thetas
def calMSE(x,y,theta):
    m= np.shape(x)[0]
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y
    cost = np.sum(loss ** 2) / (m)
    return cost
