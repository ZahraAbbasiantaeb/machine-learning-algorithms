from scipy.io import loadmat
import numpy as np
import pandas as pd
import math

A_class_index=1
B_class_index=0

#gets gaussian probability of given values
def gaussian_prob(value,mean,var):
    tmp=1/(math.sqrt(2*math.pi*var))
    power=math.exp(-(value-mean)/(2.0*var))
    return tmp*power

#gets gaussian distribution of given data
def gaussian_df(data):
    dfObj = pd.DataFrame(data)
    mean_A = []
    var_A = []
    mean_B = []
    var_B = []
    size=np.shape(dfObj)[0]
    prior_B=np.shape(dfObj[dfObj[112] == B_class_index])[0]/size
    prior_A=np.shape(dfObj[dfObj[112] == A_class_index])[0]/size
    for i in range (0,113):
        mean_A.append(np.mean(dfObj[dfObj[112] ==A_class_index][i]))
        var_A.append(np.var(dfObj[dfObj[112] ==A_class_index][i]))
        mean_B.append(np.mean(dfObj[dfObj[112] ==B_class_index][i]))
        var_B.append(np.var(dfObj[dfObj[112] ==B_class_index][i]))
    return mean_B,var_B,prior_B,mean_A,var_A,prior_A

#trains the model on train data and tests it on test data
def test_model(train,test):
    size=np.shape(test)[0]
    tag_prob=[]
    mean_B,var_B,prior_B,mean_A,var_A,prior_A=gaussian_df(train)
    tag=[]
    for row in test:
        prob_A=(prior_A)
        prob_B=(prior_B)
        t=prob_A/prob_B
        for i in range (0,112):
            A=(gaussian_prob(row[i],mean_A[i],var_A[i]))
            B=(gaussian_prob(row[i],mean_B[i],var_B[i]))
            prob_A*=A
            prob_B*=B
            t*=(A/B)
        tag_prob.append(math.log(t))
        if t>1:
            tag.append(A_class_index)
        else:
            tag.append(B_class_index)
    accuracy=0
    for i in range (0,80):
        if tag[i]==test[i,-1]:
            accuracy+=1
    return accuracy/size,tag_prob


#initialization
data = loadmat('/fMRI.mat')
data=data['Data']
test_data=[]
train_data=[]
for i in range(0,6):
    test_data.append(data[i * 80:(i + 1) * 80, :])
    dat=data
    index=[]
    for j in range(i * 80,(i + 1) * 80):
        index.append(j)
    train_data.append(np.delete(dat, index, axis=0))

