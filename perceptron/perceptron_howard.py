# -*- coding:utf-8 -*-
#/usr/bin/python 
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pyplot as pl                                                     
from matplotlib import animation as ani

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
                record.append([list(w),b])
        #if num > 0: record.append([list(w),b])
        #print 'w is %s , and bias is %s ' % (w,b)
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
        #w += eta * m[ncol-1] * m[0:ncol-1]
        b += eta * sum(M[:,ncol-1]) / len(M)
        record.append([list(w),b])
        M = get_M(w,b,data_array)
    return w,b,record
            




class showPicture:  
    def __init__(self,data,w,b):  
        self.b = b   
        self.w = w   
        plt.figure(1)  
        plt.title('Plot 1', size=14)  
        plt.xlabel('x-axis', size=14)  
        plt.ylabel('y-axis', size=14)  
        xData = np.linspace(0, 5, 100)  
        yData = self.expression(xData)  
        plt.plot(xData, yData, color='r', label='y1 data')  
        plt.scatter(data[data[:,2]==1, 0], data[data[:,2]==1, 1], color='red', marker='o')
        plt.scatter(data[data[:,2]==-1, 0], data[data[:,2]==-1, 1], color='blue', marker='x')
        plt.savefig('2d.png',dpi=75)  
    def expression(self,x):  
        y = (-self.b - self.w[0]*x)/self.w[1]    
        return y   
    def show(self):  
        plt.show()  


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
      

if __name__ == '__main__':
    data1=[[1,4,1],[0.5,2,1],[2,2.3, 1], [1, 0.5, -1], [2, 1, -1],[4,1,-1],[3.5,4,1],[3,2.2,-1]]
    #data = [[3,3,1],[4,3,1],[1,1,-1]]
    data_array = np.array(data1)
    data=[[(1,4),1],[(0.5,2),1],[(2,2.3), 1], [(1, 0.5), -1], [(2, 1), -1],[(4,1),-1],[(3.5,4),1],[(3,2.2),-1]] 
    w = np.zeros(2)
    b = 0.0
    eta = 0.5
    #w_n,b_n,record = get_w_b(w,b,eta,data_array)
    w_n,b_n,record = get_w_b_B2(w,b,eta,data_array)
    print 'record, ',record  
    print 'weight is %s bias is%s' % (w_n,b_n)
    #picture = showPicture(data_array,w=w_n,b=b_n)  
    #picture.show()  

    #display the animation of the line change in searching process  
    fig = pl.figure()  
    ax = pl.axes(xlim=(-1, 5), ylim=(-1, 5))  
    line,=ax.plot([],[],'g',lw=2)  
      
    animat=ani.FuncAnimation(fig,animate,init_func=init,frames=len(record),interval=1000,repeat=True,
                                       blit=True)  
    pl.show()  
