#-*-coding:utf-8-*-
#!/usr/bin/python
import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_cost(cost):
    plt.xlabel('iteration number')
    plt.ylabel('cost value')
    plt.title('curve of cost value')
    klen = len(cost)
    leng = np.linspace(1, klen, klen)
    plt.plot(leng, cost)
    plt.show()  

class soft_max(object):
    def __init__(self,data_arry,alpha,lam,iter_max,k,eplise):
        self.nrow,self.ncol = data_array.shape
        self.scale_X = preprocessing.scale(data_array[:,0:self.ncol-1])
        self.X = np.hstack((self.scale_X,np.repeat(1,self.nrow).reshape(self.nrow,1)))
        self.Y = data_array[:,self.ncol-1].reshape(self.nrow,1) - 1
        self.k = k
        self.Y_mat = np.zeros((self.nrow,3))
        for i in xrange(self.nrow):
            self.Y_mat[i,self.Y[i][0]] = 1.0

        np.random.seed(1)
        self.theta = np.random.random((self.k,self.ncol))/self.nrow
        self.alpha = alpha
        self.lam = lam
        self.iter_max = iter_max
        self.eplise = eplise

    def softmax_train_method1(self):
        self.costs = []
        iter = 0
        while iter < self.iter_max:
            cost_first = 0
            cost_second = 0
            ep = 0
            for j in xrange(self.k):
                Delta_j = np.zeros((1,self.ncol))
                for i in xrange(self.nrow):
                    x = self.X[i]
                    y = self.Y[i]
                    tf = get_01(y,j)
                    p = get_p(j,x,self.theta)
                    #gradient of softmax
                    Delta_j += x * (tf - p)
                    #loss of softmax
                    cost_first += tf * np.log(p)
                cost_second += np.sum(self.theta[j] ** 2)

                # L2 regulation
                Delta_j = - float(1)/self.nrow * Delta_j + self.lam * self.theta[j]
                ep += np.sum(np.fabs(Delta_j))
                # update theta
                self.theta[j] = self.theta[j] -  self.alpha * Delta_j
            cost = -cost_first/float(self.nrow) +  self.lam * cost_second / 2
            self.costs.append(cost )
            iter += 1
            if ep < self.eplise:
                return self 
        return self

    def softmax_train_method2(self):
        self.costs = []
        iter = 0
        while( iter < self.iter_max):
            exp_theta_x = np.exp(np.dot(self.X,self.theta.T))
            exp_theta_x_p = exp_theta_x / exp_theta_x.sum(axis =1).reshape(self.nrow,1)
            Delta_theta = -np.dot((self.Y_mat - exp_theta_x_p).T,self.X)/float(self.nrow)\
                         + self.lam * self.theta
            cost = -np.sum(self.Y_mat * np.log(exp_theta_x_p))/self.nrow \
                   + self.lam / 2 * np.sum(self.theta ** 2)
            cost2 = -np.sum( np.log(exp_theta_x_p[range(self.nrow),self.Y.astype(np.int).reshape((self.nrow,))]))/self.nrow \
                   + self.lam / 2 * np.sum(self.theta ** 2)

            self.theta -= self.alpha * Delta_theta
          
            self.costs.append(cost)
            iter += 1
            if np.sum(np.fabs(Delta_theta)) < self.eplise:
                return self
        return self

if __name__ == '__main__':
    data = pd.read_csv('iris_softmax.csv')
    #data = pd.read_csv('http://oheum0xlq.bkt.clouddn.com/iris_softmax.csv')
    data_array = np.array(data)
    k = 3
    alpha = 0.5 
    lam = 0.1
    iter_max = 100
    eplise = 0.01
    soft = soft_max(data_array,alpha,lam,iter_max,k,eplise)
    #a = soft.softmax_train_method1()
    b = soft.softmax_train_method2()
    #p_y = np.dot(a.X,a.theta.T)
    p_y = np.dot(b.X,b.theta.T)
    print np.argmax(p_y,axis=1)
    plot_cost(b.costs)



