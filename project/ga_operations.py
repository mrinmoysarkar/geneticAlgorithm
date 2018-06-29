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
from pgmpy.estimators import K2Score, BicScore, BdeuScore, HillClimbSearch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graph_tool.all import *
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator
import time
import sys
from multiprocessing import Process, Queue

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

def getBestStructure(pop,popSize,var,scoreFunc):
    score=np.zeros(popSize)
    for i in range(popSize):
        #model = get_structure_from_string(pop[i],var)
        #score[i] = scoreFunc.score(model)
        score[i] = scoreFunc.score(pop[i])
    return pop[np.argmax(score)], np.max(score)

def selection(pop,popSize,var,score):
    # score=np.zeros(popSize)
    # for i in range(popSize):
    #     #model = get_structure_from_string(pop[i],var)
    #     #score[i] = scoreFunc.score(model)
    #     score[i] = scoreFunc.score(pop[i])
    popScore = np.copy(score)
    avgScore = np.mean(score)
    noOfsamples = 5;
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
    return p1,p2,avgScore,popScore
        
#def replace(pop,child1,child2,popScore):
#    i1 = np.argmin(popScore)
#    pop[i1] = child1
#    popScore[i1] = 100
#    i2 = np.argmin(popScore)
#    pop[i2] = child2
#    return pop
    
def crossover(pop,p1,p2,pc,stringLen):
    if np.random.rand() > pc:
        crossSite = np.random.randint(low=1,high=stringLen-2)
        temp = pop[p1,:]
        child1 = np.zeros(stringLen)
        child2 = np.zeros(stringLen)
        child1[0:crossSite] = pop[p1,0:crossSite]
        child1[crossSite:stringLen] = pop[p2,crossSite:stringLen]
        child2[0:crossSite] = pop[p2,0:crossSite]
        child2[crossSite:stringLen] = pop[p1,crossSite:stringLen]
#        pop[p1,crossSite:stringLen] = pop[p2,crossSite:stringLen]
#        pop[p2,crossSite:stringLen] = temp[crossSite:stringLen]
        return child1, child2
    return [],[]

#def mutation(pop,pm,popSize,strigLen):
#    for i in range(popSize):
#        for j in range(stringLen):
#            if np.random.rand() > pm:
#                if pop[i,j] == 1: 
#                    pop[i,j] = 0;
#                else:
#                    pop[i,j] = 1;
#    return pop
def mutation(child,pm,stringLen):
    for j in range(stringLen):
        if np.random.rand() > pm:
            if child[j] == 1: 
                child[j] = 0;
            else:
                child[j] = 1;
    return child

def replace(pop,popSize,stringLen,score):
    if pop.shape[0] == popSize:
        return pop
    # score=np.zeros(pop.shape[0])
    # for i in range(pop.shape[0]):
    #     #model = get_structure_from_string(pop[i],var)
    #     #score[i] = scoreFunc.score(model)
    #     score[i] = scoreFunc.score(pop[i])
    newpop = np.zeros([popSize,stringLen])
    for i in range(popSize):
        i1 = np.argmax(score)
        score = np.delete(score,i1)
        newpop[i] = pop[i1]
        pop = np.delete(pop,i1,axis=0)
    return newpop
    
def reGenerate(pop,popSize,var,score,scoreFunc,stringLen,pc,pm):
    tempop = np.copy(pop)
    for i in range(int(popSize/2)):
        p1,p2,avgScore,popScore = selection(pop,popSize,var,score)
        child1,child2 = crossover(pop,p1,p2,pc,stringLen)
        if (len(child1) != 0) and (len(child2) != 0):
            child1 = mutation(child1,pm,stringLen)
            child2 = mutation(child2,pm,stringLen)
            score.append(scoreFunc.score(child1))
            tempop = np.append(tempop,[child1],axis=0)
            score.append(scoreFunc.score(child2))
            tempop = np.append(tempop,[child2],axis=0)
    pop = replace(tempop,popSize,stringLen,score)
    return pop


def getStatisticOfGen(pop,popSize,scoreFunc):
    score=np.zeros(pop.shape[0])
    for i in range(pop.shape[0]):
        #model = get_structure_from_string(pop[i],var)
        #score[i] = scoreFunc.score(model)
        score[i] = scoreFunc.score(pop[i])      
    return score, np.max(score), np.mean(score)
        
def plotGraph(string,var,fileName):
    g = Graph()
    vlist = []
    for i in range(len(var)):
        v = g.add_vertex()
        vlist.append(v)
    elist = []
    indx = 0
    for i in range(len(var)-1):
        for j in range(i+1):
            if string[indx] == 1:
                e = g.add_edge(vlist[j], vlist[i+1])
                elist.append(e)
            indx += 1;
    vprop = g.new_vertex_property("string")
    for i in range(len(var)):
        vprop[vlist[i]] = var[i]
    g.vertex_properties["name"]=vprop 
    graph_draw(g, 
               vertex_text=g.vertex_properties["name"], 
               vertex_font_size=18,
               output=fileName)#,
               #output_size=(1300, 1300))
               
def readGroundTruth(fileName):
    fileName = 'ground_truth_' + fileName
    f = open(fileName,'r')
    string = f.read()
    struc = []
    for i in range(len(string)-1):
        struc.append(float(string[i]))
    return struc
    


def getFactorVal(var, values, evidence):
    values = values[evidence[var[0]]]
    del(var[0])
    if len(var)==0:
        return values
    else:
        return getFactorVal(var,values,evidence)
        
def getjointProbability(parameters, evidence, var):
    #factors = inference.factors
    jointProb = 1;
    for i in range(len(parameters)):
        #print(var[i])
        fact = parameters[i]
        #print(fact)
        variables = list(fact.variables)
        values = list(fact.values)
        factVal = getFactorVal(variables, values, evidence)
        jointProb *= factVal
        #print(factVal)
    return jointProb
    
def getLoglikelihood(string, var, dataSet):
    dataInDict = dataSet.to_dict('index')
    model = get_structure_from_string(string,var)
    estimator = BayesianEstimator(model, dataSet)
    parameters = estimator.get_parameters()
    n = len(dataInDict)
    total = 0
    for i in range(n):
        evidence = dataInDict[i]        
        jp = getjointProbability(parameters, evidence,var)
        jp = -np.math.log10(jp)
        total += jp
    #total = -np.math.log10(total/n)

    #print('jp:',total)
    return total
    
def runGA(noOfgeneration,popSize,stringLen,var,pc,pm,scoreFunc,q):
    scoresAvg = np.zeros(noOfgeneration)
    scoresMax = np.zeros(noOfgeneration)
    pop = generate_pop(popSize,stringLen)
    popScore,maxScore,avgScore = getStatisticOfGen(pop,popSize,scoreFunc)
    scoresAvg[0] += avgScore
    scoresMax[0] += maxScore   
    for i in range(noOfgeneration-1):
        print("regenerate start")
        pop = reGenerate(pop,popSize,var,popScore,scoreFunc,stringLen,pc,pm)
        print("regenerate complete")
        popScore,maxScore,avgScore = getStatisticOfGen(pop,popSize,scoreFunc)
        scoresAvg[i+1] += avgScore
        scoresMax[i+1] += maxScore
        print("generation: ",(i+1))
    #q.put(scoresAvg) 
    #plt.plot(scoresAvg)
    #plt.show()
    return pop