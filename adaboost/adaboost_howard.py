#-*-coding:utf-8-*-
#/usr/bin/python
import pandas as pd
import numpy as np


def get_error(data_array,col,value,D):
    m,n = data_array.shape

    # greater than the value we let the y to 1
    results = np.ones(m)-2
    results[data_array[:,col] >= value] = 1
    error_gt_D = np.sum(np.fabs(data_array[:,-1] - results) * D)/2 
    error_gt = np.fabs(data_array[:,-1] - results)/2.0

    # less than the value we let the y to 1 
    results = np.ones(m)-2
    results[data_array[:,col] < value] = 1
    error_lt_D = np.sum(np.fabs(data_array[:,-1] - results) * D)/2
    error_lt = np.fabs(data_array[:,-1] - results)/2.0
    #print 'col, %s value: %s result_gt:_d %s result_lt_d: %s' % (col,value,error_gt_D,error_lt_D)
    if error_gt_D < error_lt_D:
        error_D = error_gt_D
        g_or_l = 'gt'
        error = error_gt
    else:
        error_D = error_lt_D
        g_or_l = 'lt'
        error = error_lt
    return error_D,g_or_l,error


def get_stump(data_array,D):
    min_error = np.inf
    m,n = data_array.shape
    numSteps = 10.0;
    for col in xrange(n-1):
        values = set(data_array[:,col])
        rangeMin = data_array[:,col].min(); rangeMax = data_array[:,col].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in xrange(-1,int(numSteps)+1):
            value = (rangeMin + float(j) * stepSize)
            error_D,g_or_l,error = get_error(data_array,col,value,D)
            #print 'col: %s, value: %s ' % (col,value) 
            #print 'error_d: ',error_D
            if error_D < min_error:
                min_error = error_D 
                best_col = col
                best_value = value
                best_gl = g_or_l
                best_error = error.copy()
    return (best_col,best_value,best_gl),min_error,best_error

def get_classifier(data_array, T = 50):
    m,n = data_array.shape     
    D = np.repeat(1.0/m, m)
    a = []
    h = []
    for t in xrange(T):
        h_t,e_D_t,e = get_stump(data_array,D)
        a_t = 0.5 * np.log((1-e_D_t)/e_D_t)
        #if e_t > 0.5 :
        #   h.append(h_t)
        #   a.append(a_t)
        print 'h: %s, a: %s' % (h_t,a_t)
        h.append(h_t)
        a.append(a_t)
        e[e==0] = -1
        z = D * np.exp(a_t * e) 
        D = z / np.sum(z) 
    return h,a

def get_predict(data_test,h,a):
    predict = []
    for row in data_test:
        num_stump = len(h)
        totol = 0
        for i in xrange(num_stump):
            col = h[i][0]; value = h[i][1]; gl = h[i][2]
            #print 'row[col] %s and value is %s' % (row[col],value)
            if (gl == 'gt' and row[col] >= value) or (gl == 'lt' and row[col] < value):
                pre = 1
            else:
                pre = -1
            totol += a[i] * pre
        predict.append(np.sign(totol))
    accuracy = 1 - np.sum(np.fabs(data_test[:,-1] - predict))/2/len(predict)
    return accuracy

data_df = pd.read_csv('horseColicTraining2.csv')
data_array = np.array(data_df)
h,a = get_classifier(data_array)

test = pd.read_csv('horseColicTest2.csv')
data_test = np.array(test)
accuracy = get_predict(data_test,h,a)
print 'accuracy: ',accuracy











