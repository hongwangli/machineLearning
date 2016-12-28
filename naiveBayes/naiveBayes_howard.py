#coding = utf-8
import numpy as np
import pandas as pd

def get_word_set(postingList):
    word_set = set([])
    for i in xrange(len(postingList)):
        s = set(postingList[i])
        word_set.update(s)
    return list(word_set)

def get_word_matrix(word_set,postingList):
    data_word = np.zeros((len(postingList),len(word_set)))
    for i in xrange(len(postingList)):
        count_word = [postingList[i].count(s) for s in word_set]
        data_word[i] = count_word
    data_word = pd.DataFrame(data_word,columns = word_set)
    return data_word

def get_p(data_word,classVec):
    class_set = list(set(classVec))
    len_class = len(class_set)
    p_matrix = np.zeros((len_class,data_word.shape[1]))
    for i in xrange(len_class):
        p_matrix[i] =  np.sum(data_word[np.array(classVec) ==class_set[i] ],axis =0)/np.sum(data_word,axis=0)
    return pd.DataFrame(p_matrix,index = class_set,columns =data_word.columns) 


def get_class(testEntry,p_matrix,classVec):
    class_set = list(set(classVec))
    p = -1
    for i in xrange(len(class_set)):
        p_i = float(classVec.count(class_set[i]))/len(classVec)
        p_word = [p_matrix.loc[class_set[i],word] for word in testEntry if word in word_set]
        p_i_word = p_i * reduce(lambda x,y : x*y,p_word)
        print 'class: %s p_i_word %s' % (class_set[i],p_i_word)
        if p_i_word > p : 
           p = p_i_word
           class_predict = class_set[i]
    return p,class_predict

if __name__ == '__main__':
   postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], 
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], 
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], 
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'], 
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], 
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']] 
   classVec = [0,1,0,1,0,1]  
   word_set = get_word_set(postingList)
   data_word = get_word_matrix(word_set,postingList)
   p_matrix = get_p(data_word,classVec)
   testEntry = ['love', 'my', 'dalmation']
   p,class_predict = get_class(testEntry,p_matrix,classVec)
   #test2 = ['stupid', 'garbage']
   #p,class_predict = get_class(test2,p_matrix,classVec)
   print 'p:',p
   print 'class_predict:',class_predict


