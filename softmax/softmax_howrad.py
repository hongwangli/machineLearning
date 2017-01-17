#-*-coding:utf-8-*-
#!/usr/bin/python
import numpy as np
from sklearn import preprocessing
import pandas as pd


def get_p(j,x,theta):
    kk,_ = theta.shape
    total = 0
    for l in xrange(kk):
        total += np.exp(np.dot(theta[l].T,x))
    p = np.exp(np.dot(theta[j].T,x))/total
    return p 

def get_01(y,j):
    if y == j:
       return 1
    else:
       return 0


class soft_max(object):
    def __init__(self,X,Y,Y_mat,theta,alpha,lam,iter_max,k):
        self.X = X
        self.Y = Y
        self.Y_mat = Y_mat
        self.theta = theta
        self.alpha = alpha
        self.lam = lam
        self.iter_max = iter_max
        self.k = k
        self.nrow,self.ncol = X.shape
    def softmax_train_method1(self):
        nrow,ncoll = self.X.shape
        for _ in xrange(self.iter_max):
            for j in xrange(self.k):
                Delta_j = np.zeros((1,self.ncol))
                for i in xrange(nrow):
                    x = self.X[i]
                    y = self.Y[i]
                    Delta_j += x * (get_01(y,j) - get_p(j,x,self.theta))
                # L2 regulation
                Delta_j = - float(1)/self.nrow * Delta_j + self.lam * self.theta[j]
                self.theta[j] = self.theta[j] -  self.alpha * Delta_j
        return self

    def softmax_train_method2(self):
        for _ in xrange(self.iter_max):
            exp_theta_x = np.exp(np.dot(self.X,self.theta.T))
            exp_theta_x_p = exp_theta_x / exp_theta_x.sum(axis =1).reshape(self.nrow,1)
            Delta_theta = -np.dot((self.Y_mat - exp_theta_x_p).T,self.X)/float(self.nrow)\
                         + self.lam * self.theta
            self.theta -= self.alpha * Delta_theta
        return self

if __name__ == '__main__':

    data = pd.read_csv('iris_softmax.csv')

    #data = pd.read_csv('http://oheum0xlq.bkt.clouddn.com/iris_softmax.csv')

    data_array = np.array(data)

    nrow,ncol = data_array.shape

    scale_X = preprocessing.scale(data_array[:,0:ncol-1])

    X = np.hstack((scale_X,np.repeat(1,nrow).reshape(nrow,1)))

    Y = data_array[:,ncol-1].reshape(nrow,1) - 1

    k = 3

    Y_mat = np.zeros((nrow,3))
    for i in xrange(nrow):
        Y_mat[i,Y[i][0]] = 1.0

    np.random.seed(1)
    theta = np.random.random((k,ncol))/nrow
    alpha = 0.5 
    lam = 0.1
    iter_max = 10
    soft = soft_max(X,Y,Y_mat,theta,alpha,lam,iter_max,k)
    a = soft.softmax_train_method1()
    b = soft.softmax_train_method2()
    p_y = np.dot(X,a.theta.T)
    np.argmax(p_y,axis=1)





