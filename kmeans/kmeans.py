
# -*- coding:utf-8 -*-
#/usr/bin/python

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt  

def get_distance(vector1,vector2):
    distance = np.sqrt(np.sum((vector1 - vector2)**2))
    return distance

def reCenter(data_result,k):
    centers = []
    nrow,ncol = data_result.shape
    for j in xrange(k):                                                                                   
        data_cluster = data_result[data_result[:,ncol-1]==j]
        center = np.mean(data_cluster,axis=0)  
        centers.append(center[0:ncol-1])
    return centers

def assign(centers,data_result,k):
    nrow,ncol = data_result.shape
    costs = 0
    for i in xrange(nrow):
        distances = []
        for j in xrange(k):
            distance = get_distance(data_result[i,0:ncol-1],centers[j])
            distances.append(distance)
        cost = min(distances)
        costs += cost     
        min_index = distances.index(cost)
        data_result[i,ncol-1] = min_index        
    return data_result,costs

def kmeans(data_result,centers,k,eplise):
    loop = 0
    costs = 0
    while (loop < iter_max):
        cost_old = costs
        data_result,costs = assign(centers,data_result,k)
        centers = reCenter(data_result,k)
        if (np.fabs(cost_old - costs) < eplise):
            return data_result,costs,centers
        print 'costs',costs
        loop += 1 
    return data_result,costs,centers

def showCluster(data_result,k,centers):
    nrow,ncol = data_result.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    for i in xrange(nrow):  
        markIndex = int(data_result[i,ncol-1])
        plt.plot(data_result[i,0], data_result[i,1], mark[markIndex])  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb'] 
    for i in range(k):                                                                                    
        plt.plot(centers[i][0], centers[i][1], mark[i], markersize = 12) 
    plt.show()


def find_best_k(data_result,K,eplise):
    costs = []
    nrow,ncol = data_result.shape
    for k in xrange(2,K):
        print 'k',k
        centers = [data_result[random.randint(0,nrow-1),0: ncol-1] for i in xrange(k)]
        _,cost,_ = kmeans(data_result,centers,k,eplise)
        costs.append(cost)
    return costs


def plot_cost(cost):
    plt.xlabel('iteration number')
    plt.ylabel('cost value')
    plt.title('curve of cost value')
    klen = len(cost)
    leng = np.linspace(1, klen, klen)
    plt.plot(leng, cost)
    plt.show()    

if __name__ == '__main__':
    K = 10
    iter_max = 10
    eplise = 0.001

    data  = pd.read_csv('/root/Desktop/machineLearning/kmeans/kmeans_test.csv')
    data_array = np.array(data)
    nrow,ncol = data_array.shape
    data_result = np.hstack((data_array,np.zeros(nrow).reshape(nrow,1)))

    costs = find_best_k(data_result,K,eplise)
    plot_cost(costs)

    #k =  4
    #centers = [data_result[random.randint(0,nrow-1),0: ncol] for i in xrange(k)]
    #data_result,costs,centers = kmeans(data_result,centers,k,eplise)
    #showCluster(data_result,k,centers)


