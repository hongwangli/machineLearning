#-*-coding:utf-8-*-
#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import datasets 
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

class ann_bp(object):
    def __init__(self,data_array,num_2,num_3,alpha,reg_lambda,eplise,iter_max):
        np.random.seed(1)
        self.nrow,self.ncol = data_array.shape
        self.scale_X = preprocessing.scale(data_array[:,0:self.ncol-1])
        self.W1 = np.random.random(size=(self.ncol-1,num_2)) 
        self.b1 = np.random.random(size=(1,num_2))
        self.W2 = np.random.random(size=(num_2,num_3))
        self.b2 = np.random.random(size=(1,num_3))
        self.X = np.hstack((self.scale_X,np.repeat(1,self.nrow).reshape(self.nrow,1)))
        self.Y = data_array[:,self.ncol-1].reshape(self.nrow,1)
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.eplise = eplise
        self.iter_max = iter_max
    def get_arg_sigmoid(self):
        self.costJs =[]
        iter = 0
        while iter < self.iter_max:
            
            W_b_1 = np.vstack((self.W1,self.b1))
            z2 = np.dot(self.X,W_b_1)
            a2 = sigmoid(z2)
            a2_1 = np.hstack((a2,np.repeat(1,self.nrow).reshape(self.nrow,1)))
            
            W_b_2 = np.vstack((self.W2,self.b2))
            z3 = np.dot(a2_1,W_b_2)
            a3 = sigmoid(z3)
            self.a3 = a3
            J = np.sum((self.Y-a3)**2)
            self.costJs.append(J)
            
            delta_3 = -np.multiply((self.Y - a3),sigmoid_derivative(z3))
            delta_2 = np.multiply(np.dot(delta_3,self.W2.T) ,sigmoid_derivative(z2))
            
            Delta_W_2_J = np.dot(a2.T, delta_3) 
            Delta_W_1_J = np.dot(self.scale_X.T,delta_2)
            Delta_b_2_J = delta_3.sum(axis = 0)
            Delta_b_1_J = delta_2.sum(axis = 0)
        
            Delta_W_2_J += self.reg_lambda * self.W2
            Delta_W_1_J += self.reg_lambda * self.W1
        
            self.W2 += -self.alpha * Delta_W_2_J
            self.W1 += -self.alpha * Delta_W_1_J
            self.b1 += -self.alpha * Delta_b_1_J
            self.b2 += -self.alpha * Delta_b_2_J
            ep = np.sum(np.fabs(Delta_W_2_J)) + np.sum(np.fabs(Delta_W_1_J)) \
                +np.sum(np.fabs(Delta_b_2_J)) + np.sum(np.fabs(Delta_b_1_J))
            if ep < self.eplise:
               return self
        return self

    def get_arg_softmax(self):
        self.costJs =[]
        iter = 0
        while iter < self.iter_max:
            
            W_b_1 = np.vstack((self.W1,self.b1))
            z2 = np.dot(self.X,W_b_1)
            a2 = sigmoid(z2)
            a2_1 = np.hstack((a2,np.repeat(1,self.nrow).reshape(self.nrow,1)))
            
            W_b_2 = np.vstack((self.W2,self.b2))
            z3 = np.dot(a2_1,W_b_2)
            a3 = np.exp(z3)/np.exp(z3).sum(axis=1).reshape(self.nrow,1)
            self.a3 = a3
            idx_col = self.Y.astype(np.int).reshape((self.nrow,))
            idx_row = range(self.nrow)
            J = -np.sum(np.log(a3[idx_row,idx_col]))/float(self.nrow)
            self.costJs.append(J)
            delta_3 = a3
            delta_3[idx_row,idx_col] -= 1
            delta_2 = np.multiply(np.dot(delta_3,self.W2.T) ,sigmoid_derivative(z2))
            
            Delta_W_2_J = np.dot(a2.T, delta_3) 
            Delta_W_1_J = np.dot(self.scale_X.T,delta_2)
            Delta_b_2_J = delta_3.sum(axis = 0)
            Delta_b_1_J = delta_2.sum(axis = 0)
        
            Delta_W_2_J += self.reg_lambda * self.W2
            Delta_W_1_J += self.reg_lambda * self.W1
        
            self.W2 += -self.alpha * Delta_W_2_J
            self.W1 += -self.alpha * Delta_W_1_J
            self.b1 += -self.alpha * Delta_b_1_J
            self.b2 += -self.alpha * Delta_b_2_J
            ep = np.sum(np.fabs(Delta_W_2_J)) + np.sum(np.fabs(Delta_W_1_J)) \
                +np.sum(np.fabs(Delta_b_2_J)) + np.sum(np.fabs(Delta_b_1_J))
            if ep < self.eplise:
               return self
        return self


if __name__ == '__main__':
    
    alpha = 0.1
    reg_lambda = 0.3
    data = datasets.load_iris()   
    data_array = np.hstack((data['data'],data['target'].reshape(-1,1)))
    data_array[data_array[:,-1]==2,-1] = 1
    num_2 = 3
    num_3 = 2
    iter_max = 10
    eplise = 0.1
    ann_o = ann_bp(data_array,num_2,num_3,alpha,reg_lambda,eplise,iter_max)
    # if choose sigmoid please let num_3 = 1, softmax let num_3 =2
    ann_o.get_arg_softmax()
    plot_cost(ann_o.costJs)
    print ann_o.a3.argmax(axis=1)
    
    d = datasets.load_diabetes() #for regression
    data_array = np.hstack((d['data'],d['target'].reshape(-1,1)))
    ann_r = ann_bp(data_array,num_2,num_3,alpha,reg_lambda,eplise,iter_max)
    ann_r.get_arg_sigmoid()


