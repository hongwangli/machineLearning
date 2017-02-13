#-*-coding:utf-8 -*-
#from __future__ import division
from sklearn.datasets import load_iris
import numpy as np
import math
import random
class node:
    def __init__(self, col=-1, value=None, results=None, trueBranch=None, falseBranch=None):
        self.col = col
        self.value = value
        self.results = results
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        
    def getLabel(self):
        if self.results == None:
            return None
        else:
            max_counts = 0
            for key in self.results.keys():
                if self.results[key] > max_counts:
                    label = key
                    max_counts = self.results[key]
        return label

class RandomForestsClassifier:
    def __init__(self, n_bootstrapSamples=20):
        self.n_bootstrapSamples = n_bootstrapSamples
        self.list_tree = []
        self.list_random_k = []
        
    def divideSet(self, samples, column, value):
        splitFunction = None
        if isinstance(value,int) or isinstance(value,float):
            splitFunction = lambda row: row[column] >= value
        else:
            splitFunction = lambda row: row[column] == value
        set1 = [row for row in samples if splitFunction(row)]
        set2 = [row for row in samples if not splitFunction(row)]
        return (set1,set2)
    
    def uniqueCounts(self, samples):
        results = {}
        for row in samples:
            r = row[len(row)-1]
            if r not in results:
                results[r] = 0
            results[r] = results[r]+1
        return results
    
    def giniEstimate(self, samples):
        if len(samples)==0: return 0
        total = len(samples)
        counts = self.uniqueCounts(samples)
        gini = 0
        for target in counts:
            gini = gini + pow(counts[target],2)
        gini = 1 - gini / pow(total,2)
        return gini
    
    def buildTree(self, samples):#构造CART决策树
        if len(samples) == 0:
            return node()
        currentGini = self.giniEstimate(samples)
        bestGain = 0
        bestCriteria = None
        bestSets = None
        colCount = len(samples[0]) - 1
        colRange = range(0,colCount)
        np.random.shuffle(colRange)
        for col in colRange[0:int(math.ceil(math.sqrt(colCount)))]:
            colValues = {}
            for row in samples:
                colValues[row[col]] = 1
            for value in colValues.keys():
                (set1,set2) = self.divideSet(samples,col,value)
                gain = currentGini - (len(set1)*self.giniEstimate(set1) + len(set2)*self.giniEstimate(set2)) / len(samples)
                if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                    bestGain = gain
                    bestCriteria = (col,value)
                    bestSets = (set1,set2)
        if bestGain > 0:
            trueBranch = self.buildTree(bestSets[0])
            falseBranch = self.buildTree(bestSets[1])
            return node(col=bestCriteria[0],value=bestCriteria[1],trueBranch=trueBranch,falseBranch=falseBranch)
        else:
            return node(results=self.uniqueCounts(samples))
        
    def printTree(self, tree,indent='  '):#以文本形式显示决策树
        if tree.results != None:
            print str(tree.results)
        else:
            print str(tree.col)+':'+str(tree.value)+'?'
            print indent+'T->',
            self.printTree(tree.trueBranch,indent+'  ')
            print indent+'F->',
            self.printTree(tree.falseBranch,indent+'  ')
        
    def predict_tree(self, observation, tree,random_k):#利用决策树进行分类
        if tree.results != None:
            return tree.getLabel()
        else:
            v = observation[random_k[tree.col]]
            branch = None
            if isinstance(v,int) or isinstance(v,float):
                if v >= tree.value: branch = tree.trueBranch
                else: branch = tree.falseBranch
            else:
                if v == tree.value: branch = tree.trueBranch
                else: branch = tree.falseBranch
            return self.predict_tree(observation,branch,random_k)
        
    #def generateBootstrapSamples(self, data):#构造bootstrap样本
    #    samples = []
    #    for i in range(len(data)):
    #        samples.append(data[np.random.randint(len(data))])
    #    return samples
    
    def generateBootstrapSamples(self, data):#构造bootstrap样本
        k = int(np.log2(len(data[0])-1))
        samples = []
        random_k = random.sample(range(len(data[0])-1), k)
        random_k.append(len(data[0])-1)
        for i in range(len(data)):
            data1 = data[np.random.randint(len(data))]
            samples.append([data1[i] for i in random_k])
        return samples,random_k

    def fit(self, data):#构造随机森林
        for i in range(self.n_bootstrapSamples):
            samples,random_k = self.generateBootstrapSamples(data)
            currentTree = self.buildTree(samples)
            self.list_tree.append(currentTree)
            self.list_random_k.append(random_k)

    def predict_randomForests(self, observation):#利用随机森林对给定观测数据进行分类
        results = {}
        for i in range(len(self.list_tree)):
            currentResult = self.predict_tree(observation, self.list_tree[i],self.list_random_k[i])
            if currentResult not in results:
                results[currentResult] = 0
            results[currentResult] = results[currentResult] + 1
        max_counts = 0
        for key in results.keys():
            if results[key] > max_counts:
                finalResult = key
                max_counts = results[key]
        return finalResult
if __name__ == '__main__':
    #import randomforest
    iris = load_iris()
    X = iris.data
    y = iris.target
    temp_data = np.concatenate([X, y.reshape((150,1))], axis=1)
    #由于上述代码要求输入的观测数据存储在二维列表中，需将numpy二维数组转换成列表
    
    data = []
    for i in range(temp_data.shape[0]):
        temp = []
        for j in range(temp_data.shape[1]):
            temp.append(temp_data[i][j])
        data.append(temp)
    
    rowRange = range(150)
    np.random.shuffle(rowRange)
    #从鸢尾花数据集(容量为150)按照随机均匀抽样的原则选取70%的数据作为训练数据
    training_data = [data[i] for i in rowRange[0:105]]
    #按照随机均匀抽样的原则选取30%的数据作为检验数据
    testing_data = [data[i] for i in rowRange[105:150]]
    classifier = RandomForestsClassifier(n_bootstrapSamples=10)#初始化随机森林
    #classifier.generateBootstrapSamples(training_data)
    classifier.fit(training_data)#利用训练数据进行拟合
    
    finalResults = []
    for row in testing_data:
        finalResult = classifier.predict_randomForests(row[0:4])#对检验数据集进行分类
        finalResults.append(finalResult)
    errorVector = np.zeros((45,1))
    errorVector[np.array(finalResults) != (np.array(testing_data))[:,4]] =1
    print 'errot rate:', errorVector.sum()/45#计算错判率


