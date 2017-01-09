# -*- coding:utf-8 -*-
#/usr/bin/python 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as pl                                                     
from matplotlib import animation as ani
from matplotlib.colors import ListedColormap
from sklearn import preprocessing

def get_w_b(w,b,eta,data_array):
    nrow,ncol = data_array.shape
    num = 2
    record = [[list(w),b]]
    while (num > 0):
        num = 0
        for data in data_array:
            u = np.dot(data[0:ncol-1],w) + b 
            if u* data[ncol-1] <= 0:
                w += eta * data[ncol-1] * data[0:ncol-1]
                b += eta * data[ncol-1]
                num += 1
            print 'num ',num
            record.append([list(w),b])
        #if num > 0: record.append([list(w),b])
        print 'w is %s , and bias is %s ' % (w,b)
    #print record
    return w,b,record    

def get_M(w,b,data_array):
    nrow,ncol = data_array.shape
    u = np.dot(data_array[:,0:ncol-1],w) + b    
    y_u = data_array[:,ncol-1] * u
    M = data_array[y_u<=0,:]
    return M    

def get_w_b_B(w,b,eta,data_array):
    nrow,ncol = data_array.shape
    M = get_M(w,b,data_array)
    #print 'M',M
    #len_M = len(M)
    record = []
    while (len(M) > 0):
        for m in M:
            w += eta * m[ncol-1] * m[0:ncol-1]
            b += eta * m[ncol-1]
        record.append([list(w),b])
        M = get_M(w,b,data_array)
    return w,b,record


def get_w_b_B2(w,b,eta,data_array):                                                                                             
    nrow,ncol = data_array.shape
    M = get_M(w,b,data_array)
    #print 'M',M
    #len_M = len(M)
    record = []
    while (len(M) > 0): 
        w += np.dot(eta * M[:,ncol-1],M[:,0:ncol-1])/len(M)
        b += eta * sum(M[:,ncol-1]) / len(M)
        record.append([list(w),b])
        M = get_M(w,b,data_array)
    return w,b,record

def predict(X,W):
    nrow,ncol = X.shape
    X_1 = np.hstack((np.repeat(1,nrow).reshape((nrow,1)),X)) 
    y_hat = np.where(np.dot(X_1,W) <= 0 ,-1,1)
    return y_hat


class perceptron_BGD(object):
    def __init__(self,max_iter,eta):
        self.max_iter = max_iter
        self.eta = eta

    def BGD(self,w,b,data_array):
        self.data_array = data_array
        self.W = np.hstack((w,np.array(b)))
        self.record = []
        nrow,ncol = data_array.shape
        print 'nrow is %s ncol is %s' % (nrow,ncol)
        X = np.hstack((np.repeat(1,nrow).reshape((nrow,1)),data_array[:,0:ncol-1]))
        y = data_array[:,ncol-1]
        iter = 0
        while (iter < self.max_iter):
            print 'self.W ',self.W
            #y_hat = np.where(np.dot(X,self.W) >= 0,1,-1)
            y_hat = np.dot(X,self.W)
            #print 'y_hat ',y_hat
            #print 'errors ',y - y_hat
            W_old = self.W
            self.W += self.eta * np.dot(np.transpose(X),(y-y_hat))/nrow
            self.record.append([list(self.W[0:-1]),self.W[-1]])
            #if sum(abs(self.W - W_old)) < 0.001:
            #    return self
            #print 'self.W',self.W 
            iter += 1 
        return self

def plot_decision_regions(X, y, W , res=0.02):
    markers = ('s', 'o', 'x', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    colormap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res), 
                        np.arange(x2_min, x2_max, res))
    Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T,W)
    # xx1.ravel() 将xx1从numpy.narray类型的多维素组转换为一位数组
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=.4, cmap=colormap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], marker=markers[idx],\
                    alpha=.8, cmap=colormap(idx),\
                    label=np.where(cl==1, 'versicolor', 'setosa'))


#display the animation of the line change in searching process  
  
def init():
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    line.set_data([],[])  
    for p in data:  
        if p[1]>0:  
            x1.append(p[0][0])  
            y1.append(p[0][1])  
        else:  
            x2.append(p[0][0])  
            y2.append(p[0][1])  
    pl.plot(x1,y1,'or')  
    pl.plot(x2,y2,'ob')  
    return line,  
  
  
def animate(i):  
    global record,ax,line  
    w=record[i][0]  
    b=record[i][1]  
    x1=-5   
    y1=-(b+w[0]*x1)/w[1]  
    x2=6  
    y2=-(b+w[0]*x2)/w[1]  
    line.set_data([x1,x2],[y1,y2])  
    return line,  

def get_accuracy(X,W,y):
    y_hat = predict(X,W)

    

if __name__ == '__main__':

    data_df = pd.read_csv('/root/Desktop/machineLearning/perceptron/iris.csv')
    train_data = data_df.iloc[0:100,[0,2,4]]
    train_data['y'] = np.where(train_data['y']=='Iris-setosa',-1,1)
    data_array =  np.array(train_data)
    nrow,ncol = data_array.shape
    #X = np.hstack((np.repeat(1,nrow).reshape((nrow,1)),data_array[:,0:ncol-1]))
    X =  preprocessing.scale(data_array[:,0:ncol-1])
    y = data_array[:,ncol-1]
    data_array[:,0:ncol-1] = X
    w = np.zeros(2)
    b = 0.0
    eta = 0.1
    max_iter =100
    w_n,b_n,record = get_w_b(w,b,eta,data_array)
    
    #bgd_o = perceptron_BGD(max_iter,eta)                                                                             
    #bgd_o.BGD(w,b,data_array)
    #w_n = bgd_o.W[0:-1]
    #b_n = bgd_o.W[-1]

    W = np.hstack((w_n,b_n))
    plot_decision_regions(X, y, W, res=0.02)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper right')
    plt.show()


    #display the animation of the line change in searching process  
    #fig = pl.figure()  
    #ax = pl.axes(xlim=(-1, 5), ylim=(-1, 5))  
    #line,=ax.plot([],[],'g',lw=2)  
      
    #animat=ani.FuncAnimation(fig,animate,init_func=init,frames=len(record),interval=1000,repeat=True,
    #                                   blit=True)  
    #pl.show()  
