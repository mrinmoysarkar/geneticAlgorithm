# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 23:37:03 2018

@author: Mrinmoy
"""

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from  pgmpy.readwrite.BIF import BIFReader
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import K2Score, BicScore, BdeuScore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_pop(size,strLen):
    pop = np.random.rand(size,strLen)
    pop = np.round(pop)
    return pop

def get_structure_from_string(string,var):
    indx = 0
    structure = []
    for i in range(len(var)-1):
        for j in range(i+1):
            if string[indx] == 1:
                topl = (var[j],var[i+1])
                structure.append(topl)
            indx += 1;
    model = BayesianModel(structure)
    return model

def selection(pop,popSize,var,scoreFunc):
    score=np.zeros(popSize)
    for i in range(popSize):
        model = get_structure_from_string(pop[i],var)
        score[i] = scoreFunc.score(model)
    avgScore = np.mean(score)
    noOfsamples = 10;
    indexs = np.zeros(noOfsamples,dtype=int);
    for i in range(noOfsamples):
        indexs[i] = np.argmax(score)
        mi = np.min(score)
        score[indexs[i]] = mi
    i1 = np.random.randint(low=0,high=noOfsamples-1)
    p1 = indexs[i1]
    indexs = np.delete(indexs,i1);
    i2 = np.random.randint(low=0,high=noOfsamples-2)
    p2 = indexs[i2]
    return p1,p2,avgScore
        
    

def crossover(pop,p1,p2,pc,stringLen):
    crossSite = np.random.randint(low=1,high=stringLen-2)
    if np.random.rand() > pc:
        temp = pop[p1,:]
        pop[p1,crossSite:stringLen] = pop[p2,crossSite:stringLen]
        pop[p2,crossSite:stringLen] = temp[crossSite:stringLen]
    return pop

def mutation(pop,pm,popSize,strigLen):
    for i in range(popSize):
        for j in range(stringLen):
            if np.random.rand() > pm:
                if pop[i,j] == 1: 
                    pop[i,j] = 0;
                else:
                    pop[i,j] = 1;
    return pop

if __name__ == '__main__':
    bnNetworkFileName = 'asia.bif'
    bn = BIFReader(path=bnNetworkFileName)
    var = bn.get_variables()
    bnModel = bn.get_model()
    inference = BayesianModelSampling(bnModel)
    dataSet = inference.forward_sample(size=500, return_type='dataframe')
    
    model = BayesianModel()
    model.add_nodes_from(var)
        
    model.fit(dataSet, estimator=MaximumLikelihoodEstimator)
#    for i in range(len(var)):
#        print('estimated:')
#        print(model.get_cpds(var[i]))
#        print('original:')
#        print(bnModel.get_cpds(var[i]))
#    mdl = BayesianModel([('asia', 'tub'), ('tub', 'smoke'), ('tub', 'lung'), ('tub', 'bronc'),('bronc','either'),('bronc','xray'),('xray','dysp')]) 
#    
    
    
    k2 = K2Score(dataSet)
    bic = BicScore(dataSet)
    bdeu = BdeuScore(dataSet)
    
#    print(k2.score(bnModel))
#    print(k2.score(model))
#    
#    print(bic.score(bnModel))
#    print(bic.score(model))
#    
#    print(bdeu.score(bnModel))
#    print(bdeu.score(model))
    
    noOfvar = len(var)
    popSize = 100;
    noOfgeneration = 1000
    pm = 0.001
    pc = 0.6
    stringLen = int(noOfvar*(noOfvar-1)/2)
    
    pop = generate_pop(popSize,stringLen)
    #model = get_structure_from_string(pop[1],var)
    
    #print(k2.score(model))
    #model.fit(dataSet, estimator=MaximumLikelihoodEstimator)
    #print(k2.score(model))
    #crossover(pop,1,2,0.6,stringLen)
    scores = np.zeros(noOfgeneration)
    for i in range(noOfgeneration):
        p1,p2,avgScore = selection(pop,popSize,var,k2)
        scores[i] = avgScore
        pop = crossover(pop,p1,p2,pc,stringLen)
        pop = mutation(pop,pm,popSize,stringLen)
        print("generationNo.: ",i)
    #print(p1,p2,avgScore)
    plt.plot(scores)
    plt.show()