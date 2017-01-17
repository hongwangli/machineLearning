#-*-coding:utf-8-*-
#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

def sigmoid(z):
    y=1.0 / (1 + np.exp(-z))
    return y

def sigmoid_derivative(z):
    return np.multiply(sigmoid(z),(1 - sigmoid(z)))


def plot_cost(cost):
    plt.xlabel('iteration number')
    plt.ylabel('cost value')
    plt.title('curve of cost value')
    klen = len(cost)
    leng = np.linspace(1, klen, klen)
    plt.plot(leng, cost)
    plt.show()

alpha = 0.1
reg_lambda = 0.3

data = pd.read_csv('trainData.csv')
data_array = np.array(data)
nrow,ncol = data_array.shape
scale_X = preprocessing.scale(data_array[:,0:ncol-1])
X = np.hstack((scale_X,np.repeat(1,nrow).reshape(nrow,1)))
Y = data_array[:,ncol-1].reshape(nrow,1)

np.random.seed(1)
W1 = np.random.random(size=(4,3))
b1 = np.random.random(size=(1,3))
W2 = np.random.random(size=(3,1))
b2 = np.random.random(size=(1,1))

costJs =[]

for i in xrange(1000):

    W_b_1 = np.vstack((W1,b1))
    z2 = np.dot(X,W_b_1)
    a2 = sigmoid(z2)
    a2_1 = np.hstack((a2,np.repeat(1,nrow).reshape(nrow,1)))
    
    W_b_2 = np.vstack((W2,b2))
    z3 = np.dot(a2_1,W_b_2)
    a3 = sigmoid(z3)
    J = np.sum((Y-a3)**2)
    costJs.append(J)
    
    delta_3 = -np.multiply((Y - a3),sigmoid_derivative(z3))
    delta_2 = np.multiply(np.dot(delta_3,W2.T) ,sigmoid_derivative(z2))
    
    Delta_W_2_J = np.dot(a2.T, delta_3) 
    Delta_W_1_J = np.dot(scale_X.T,delta_2)
    Delta_b_2_J = delta_3.sum(axis = 0)
    Delta_b_1_J = delta_2.sum(axis = 0)

    Delta_W_2_J += reg_lambda * W2
    Delta_W_1_J += reg_lambda * W1

    W2 += -alpha * Delta_W_2_J
    W1 += -alpha * Delta_W_1_J
    b1 += -alpha * Delta_b_1_J
    b2 += -alpha * Delta_b_2_J

plot_cost(costJs)



