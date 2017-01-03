#coding=utf-8
import pandas as pd
import random 
import numpy as np

def compute_kernel_value(x,z,option):
    if option == 'linear':
       kernel_value = np.dot(x,z)
    return kernel_value


def get_u_i(alpha,b,data,test_data_row,option):
    nz_index = np.nonzero(alpha)[0].tolist()
    support_x = data.iloc[nz_index,0:ncol-1]
    support_y = data.iloc[nz_index,ncol-1]  
    support_alpha = alpha[nz_index]
    nrow_support,ncol_sport = support_x.shape
    u_i = 0
    for j in xrange(nrow_support):
        kernel_value = compute_kernel_value(test_data_row,support_x.iloc[j],option)
        u_i += support_alpha[j] * support_y.iloc[j] * kernel_value
    u_i += b
    return u_i

def get_e(alpha,b,index,data,option):
    nrow,ncol = data.shape
    test_data_row = data.iloc[index,0:ncol-1]
    u_index = get_u_i(alpha,b,data,test_data_row,option)
    y_index = data.iloc[index,ncol-1]
    e_index = u_index - y_index
    return e_index

def get_kernelMatrix(data,option):
    nrow,ncol = data.shape
    kernelMatrix = np.zeros((nrow,nrow))
    for i in xrange(nrow):
        for j in xrange(nrow):
            kernelMatrix[i,j] = compute_kernel_value(data.iloc[i,0:ncol-1],data.iloc[j,0:ncol-1],option)
    return kernelMatrix

def get_L_H(y_i,y_j,alpha_i_old,alpha_j_old,C):
    if y_i != y_j:
        L = max(0,alpha_j_old - alpha_i_old)
        H = min(C, C + alpha_j_old - alpha_i_old)
    else:
        L = max(0,alpha_j_old + alpha_i_old - C)
        H = min(C,alpha_j_old + alpha_i_old)
    return L,H

def select_index_j(index_i,alpha,b,data,error):
    alpha_i = alpha[index_i]
    nrow,ncol = data.shape
    e_i = get_e(alpha,b,index_i,data,option) 
    nz_index = np.nonzero(error[:,0])[0]
    max_error = -1
    if len(nz_index)>0:
        for j in nz_index:
            e_j = get_e(alpha,b,j,data,option)
            #print 'e_j',e_j 
            if abs(e_i - e_j) > max_error:
                max_error = abs(e_i - e_j)
                index_j = j
                e_index_j = e_j
    else :
        #print 'what is wrong'
        index_j = index_i
        while index_i == index_j:
              index_j = random.randint(0,nrow) 
        e_index_j = get_e(alpha,b,index_j,data,option)
    #print 'index_j',index_j
    return index_j,e_index_j



def update_alpha_b(index_i,alpha,b,C,toler,error,data,kernelMatrix):
    index_j,e_j = select_index_j(index_i,alpha,b,data,error) 
    alpha_i_old = alpha[index_i]

    alpha_j_old = alpha[index_j]

    nrow,ncol = data.shape
    y = data.iloc[:,ncol-1]
    y_i = y[index_i]
    y_j = y[index_j]

    u_i = get_u_i(alpha,b,data,data.iloc[index_i,0:ncol-1],option)
    u_j = get_u_i(alpha,b,data,data.iloc[index_j,0:ncol-1],option)
    e_i = u_i - y_i
    e_j = u_j - y_j
    print 'index_i is %s ,e_i is %s:' % (index_i,e_i)
    print 'index_j is %s ,e_j is %s:' % (index_j,e_j)
    ###index_j,e_j = select_index_j(index_i,alpha,b,data,error)  
    #e_i = error[index_i,1]
    #e_j = error[index_j,1] 
    #if (y_i * u_i < 1 - toler and alpha_i_old < C ) or (y_i * u_i >  1 + toler  and alpha_i_old >0 ) :
    if (y_i * e_i <  -toler) and (alpha_i_old < C ) or (y_i * e_i >  toler)  and (alpha_i_old >0 ) :
  
        eta_h = kernelMatrix[index_i,index_i] + kernelMatrix[index_j,index_j] \
              - 2 * kernelMatrix[index_i,index_j]  
        #print 'eta:---:',eta
        if eta_h < 0:
            return alpha,b,error,0        
        print 'change section:', y_j*(e_i - e_j)/eta_h
        alpha_j_new = alpha_j_old + y_j*(e_i - e_j)/eta_h
        print 'alpha_j_new is %s, alpha_j_old is %s' % (alpha_j_new,alpha_j_old)
        print 'alpha_j_old - alpha_j_new: ',alpha_j_old - alpha_j_new 
        L,H = get_L_H(y_i,y_j,alpha_i_old,alpha_j_old,C)
        if L == H:
            return alpha,b,error,0       
        print 'L is %s H is %s' % (L,H)
        if alpha_j_new > H:
           print '1111'
           alpha_j_new = H
        if alpha_j_new < L:
           print '2222'
           alpha_j_new = L

        print 'alpha_j_new: ',alpha_j_new
        if abs(alpha_j_old - alpha_j_new) < 0.00001:
            alpha[index_j] = alpha_j_new
            e_j_new = get_e(alpha,b,index_j,data,option)
            error[index_j] = [1,e_j_new]
            return alpha,b,error,0      
        alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

        b_i_new = -e_i - y_i * kernelMatrix[index_i,index_i] * (alpha_i_new - alpha_i_old) \
                  - y_j * kernelMatrix[index_j,index_i] *(alpha_j_new - alpha_j_old) + b
        b_j_new = -e_j - y_i * kernelMatrix[index_i,index_j] * (alpha_i_new - alpha_i_old) \
                  - y_j * kernelMatrix[index_j,index_j] *(alpha_j_new - alpha_j_old) + b 

        if (alpha_i_new >0 and alpha_i_new <C):
            b = b_i_new
        elif (alpha_j_new > 0 and alpha_j_new < C):
            b = b_j_new
        else:
            b = (b_i_new + b_j_new) * 0.5
        #print 'alpha_i_new',alpha_i_new
        #print 'alpha_j_new',alpha_j_new
        alpha[index_i] = alpha_i_new
        alpha[index_j] = alpha_j_new
        e_i_new = get_e(alpha,b,index_i,data,option)
        e_j_new = get_e(alpha,b,index_j,data,option) 
        error[index_i] = [1,e_i_new]
        error[index_j] = [1,e_j_new]
        return alpha,b,error,1
    else:
        print 'in the else'
        return alpha,b,error,0


def smo(iter_max,toler,data,alpha,b,error,C):
    nrow,ncol = data.shape
    all_set = True;
    iter = 0
    changed_nums = 0
    while (iter < iter_max and ( changed_nums > 0 or all_set)):
          print 'iter=============:',iter
          changed_nums = 0
          #print 'begin changed_num:',changed_nums
          if all_set :
              for index_i in xrange(nrow):
                  #index_j,e_index_j = select_index_j(index_i,alpha,b,data,error) 
                  alpha,b,error,changed_num = update_alpha_b(index_i,alpha,b,C,toler,error,data,kernelMatrix)  
                  changed_nums += changed_num
              iter += 1
              print 'all_set changed_nums:',changed_nums
          else:
              index_0_C = np.nonzero((alpha > 0) * (alpha < C) )[0]
              for index_i in index_0_C:
                  #index_j,e_index_j = select_index_j(index_i,alpha,b,data,error) 
                  alpha,b,error,changed_num = update_alpha_b(index_i,alpha,b,C,toler,error,data,kernelMatrix)
                  changed_nums += changed_num
              print 'not all set changed_nums:',changed_nums
              iter += 1
          if all_set:
              all_set = False
          elif changed_nums == 0:
              all_set = True
    return alpha,b



option = 'linear' 
data = pd.read_csv('/root/Desktop/machineLearning/svm/testSet_svm.csv')                                   
nrow,ncol = data.shape
kernelMatrix = get_kernelMatrix(data,option)
alpha = np.zeros(nrow)
C = 0.6 
b = 0
toler = 0.001
iter_max = 10
error = np.zeros((nrow,2))

a,b = smo(iter_max,toler,data,alpha,b,error,C)


#alpha[[1,10,30,50]] = [0.5,0.6,0.1,1]

#test_data = data.iloc[0:10,0:2]
#index_i = 0
#index_j,e_index_j = select_index_j(index_i,alpha,b,data)
#alpha_new,b_new = update_alpha_b(index_i,index_j,alpha,b,C,data,kernelMatrix)

#for index_i in xrange(nrow):
#    print 'index_:',index_i
#    index_j,e_index_j = select_index_j(index_i,alpha,b,data)    
#    print 'index_j is %s ,e_index_j is %s' % (index_j,e_index_j)
#    alpha,b = update_alpha_b(index_i,index_j,alpha,b,C,data,kernelMatrix)   
#    print 'alpha ',alpha
#    print 'b :',b
#
#
#
#nrow_test,ncol_test = test_data.shape
#nz_index = np.nonzero(alpha)[0].tolist()
#
#test_data = data.iloc[0:10,0:2]
#nrow_test,ncol_test = test_data.shape
#predicts = []
#for i in xrange(nrow_test):
#    test_data_row = test_data.iloc[i]
#    u_i = get_u_i(alpha,b,data,test_data_row,option)
#    predict = np.sign(u_i)
#    predicts.append(predict)
#
#print 'predict',predicts










