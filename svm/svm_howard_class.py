# -*- coding:utf-8 -*-
#! /usr/bin/python
import numpy as np 
import random 
import pandas as pd
import matplotlib.pyplot as plt

def compute_kernel_value(matrix_x,sample_x,kernelOption):                                  
    option = kernelOption[0]
    if option == 'linear':
        kernel_value = np.dot(matrix_x,sample_x)
        return kernel_value
    if option == 'rbf':
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        nrowMat,ncolMat = matrix_x.shape
        kernel_value = np.zeros(nrowMat) 
        for i in xrange(nrowMat):
            diff = matrix_x[i,:] - sample_x 
            kv = np.exp(np.dot(np.transpose(diff),diff)/(-2.0 * sigma**2))
            kernel_value[i] = kv
        return kernel_value

def calculate_kernel_matrix(matrix_x,kernelOption):
    nrow,ncol = matrix_x.shape
    kernel_matrix = np.zeros((nrow,nrow))
    for i in xrange(nrow):
        kernel_value = compute_kernel_value(matrix_x,matrix_x[i],kernelOption)
        kernel_matrix[:,i] = kernel_value
    return kernel_matrix

class svmStruct:
    def __init__(self,dataset,C,toler,kernelOption):
        self.nrow,self.ncol = dataset.shape
        self.alphas = np.zeros(self.nrow)
        self.b = 0.0
        self.errors = np.zeros((self.nrow,2))
        self.C = C
        self.toler = toler
        self.x = dataset[:,0:self.ncol-1]
        self.y = dataset[:,self.ncol-1]
        self.kernelMatrix = calculate_kernel_matrix(self.x,kernelOption)

def calculateError(svm, index_i):
    y_i = svm.y[index_i]
    u_i = sum(svm.alphas * svm.y * svm.kernelMatrix[:, index_i]) + svm.b
    e_i = u_i - y_i
    return e_i

def selectIndex_j(svm, index_i, e_i):
    svm.errors[index_i] = [1,e_i]
    nonzeros = np.nonzero(svm.errors[:,0])[0]
    maxDiff = -1; 
    if len(nonzeros) > 1:
        for index_k in nonzeros:
            if index_k == index_i:
                continue
            e_k = calculateError(svm, index_k)
            if (abs(e_k - e_i) > maxDiff):
                maxDiff = abs(e_k - e_i)
                index_j = index_k
                e_j = e_k
    else:
        index_j = index_i
        while index_j == index_i:
            index_j = random.randint(0,svm.nrow)
        e_j = calculateError(svm, index_j)
    return index_j, e_j


def updateError(svm, index_k):
    error = calculateError(svm, index_k)
    svm.errors[index_k] = [1, error]

def update_alpha_b(svm,index_i):

    e_i = calculateError(svm, index_i)
    if (svm.y[index_i] * e_i < -svm.toler) and (svm.alphas[index_i] < svm.C) or \
        (svm.y[index_i] * e_i > svm.toler) and (svm.alphas[index_i] > 0):

        index_j, e_j = selectIndex_j(svm, index_i, e_i)
        alpha_i_old = svm.alphas[index_i].copy()
        alpha_j_old = svm.alphas[index_j].copy()

        if svm.y[index_i] != svm.y[index_j]:
            L = max (0, svm.alphas[index_j] - svm.alphas[index_i])
            H = min (svm.C, svm.C + svm.alphas[index_j] - svm.alphas[index_i])
        else:
            L = max(0, svm.alphas[index_j] + svm.alphas[index_j] - svm.C)
            H = min(svm.C, svm.alphas[index_j] + svm.alphas[index_i])
        if L == H:
            return 0

        eta = 2.0 * svm.kernelMatrix[index_i,index_j] - svm.kernelMatrix[index_i,index_j]\
              -svm.kernelMatrix[index_j, index_j]
        if eta >= 0: 
            return 0

        svm.alphas[index_j] -= svm.y[index_j] * (e_i - e_j) / eta
        
        if svm.alphas[index_j] > H:
            svm.alphas[index_j] = H
        if svm.alphas[index_j] < L:
            svm.alphas[index_j] = L

        if (abs(alpha_j_old - svm.alphas[index_j]) < 0.0001 ):
            updateError(svm, index_j)
            return 0

        svm.alphas[index_i] += svm.y[index_i] * svm.y[index_j] * (alpha_j_old - svm.alphas[index_j]) 

        b1 = svm.b - e_i -svm.y[index_i] * (svm.alphas[index_i] - alpha_i_old)\
             * svm.kernelMatrix[index_i,index_i] - svm.y[index_j] * (svm.alphas[index_j] - alpha_j_old) \
             * svm.kernelMatrix[index_i,index_j]

        b2 = svm.b - e_j -svm.y[index_i] * (svm.alphas[index_i] - alpha_i_old)\
             * svm.kernelMatrix[index_i,index_j] - svm.y[index_j] * (svm.alphas[index_j] - alpha_j_old) \
             * svm.kernelMatrix[index_j,index_j]

        if (0 < svm.alphas[index_i]) and (svm.alphas[index_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[index_j]) and (svm.alphas[index_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        updateError(svm, index_j)
        updateError(svm, index_i)

        return 1

    else:
        return 0

def smo(svm,max_iter):
    iter = 0
    entireSet = True
    alpha_changed_count = 0
    while (iter < max_iter) and (alpha_changed_count > 0 or entireSet):
        alpha_changed_count = 0
        if entireSet:
            for i in xrange(svm.nrow):
                alpha_changed_count += update_alpha_b(svm, i)
            iter += 1
        else:
            nonBoundAlphaList = np.nonzero((svm.alphas > 0) * (svm.alphas < svm.C))[0]
            for i in nonBoundAlphaList:
                alpha_changed_count += update_alpha_b(svm, i)
            iter += 1
        if entireSet:
            entireSet = False
        elif alpha_changed_count == 0:
            iter += 1
    return svm 

# testing your trained svm model given test set
def testSVM(svm,dataArray_test ,kernelOption):
    nrow,ncol = dataArray_test.shape
    test_x = dataArray_test[:,0:ncol-1]
    test_y = dataArray_test[:,ncol-1]
    supportVectorsIndex = np.nonzero(svm.alphas > 0)[0]
    supportVectors      = svm.x[supportVectorsIndex]
    supportVectorLabels = svm.y[supportVectorsIndex]
    supportVectorAlphas = svm.alphas[supportVectorsIndex]
    matchCount = 0
    for i in xrange(nrow):
        kernelValue = compute_kernel_value (supportVectors, test_x[i, :] , kernelOption)
        predict = sum(kernelValue * supportVectorLabels * supportVectorAlphas) + svm.b
        if np.sign(predict) == np.sign(test_y[i]):
            matchCount += 1
    accuracy = float(matchCount) / nrow
    return accuracy

# show your trained svm model only available with 2-D data
def showSVM(svm):
    if svm.x.shape[1] != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    # draw all samples
    for i in xrange(svm.nrow):
        if svm.y[i] == -1:
            plt.plot(svm.x[i, 0], svm.x[i, 1], 'or')
        elif svm.y[i] == 1:
            plt.plot(svm.x[i, 0], svm.x[i, 1], 'ob')

    # mark support vectors
    supportVectorsIndex = np.nonzero(svm.alphas > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(svm.x[i, 0], svm.x[i, 1], 'oy')
    
    # draw the classify line
    #w = np.zeros((2, 1))
    w = [0.0,0.0]
    for i in supportVectorsIndex:
        w += np.multiply(svm.alphas[i] * svm.y[i], svm.x[i, :])
        #print np.multiply(svm.alphas[i] * svm.y[i], svm.x[i, :])
    min_x = min(svm.x[:, 0])
    max_x = max(svm.x[:, 0])
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.show()
 

if __name__=='__main__':
    data = pd.read_csv('/root/Desktop/machineLearning/svm/testSet_svm.csv')  
    #data = pd.read_csv('http://oheum0xlq.bkt.clouddn.com/testSet_svm.csv')
    dataArray_train = np.array(data)[0:81,:]
    dataArray_test = np.array(data)[80:,:]
    C = 0.6
    toler = 0.001
    kernelOption = ['linear',2]
    svm =  svmStruct(dataArray_train,C,toler,kernelOption)
    max_iter = 10
    ss = smo(svm,max_iter)
    accuracy = testSVM(ss, dataArray_test, kernelOption)
    print 'accuracy is :',accuracy
    showSVM(ss)









