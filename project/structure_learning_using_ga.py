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
        model = get_structure_from_string(pop[i],var)
        score[i] = scoreFunc.score(model)
    return pop[np.argmax(score)], np.max(score)

def selection(pop,popSize,var,scoreFunc):
    score=np.zeros(popSize)
    for i in range(popSize):
        model = get_structure_from_string(pop[i],var)
        score[i] = scoreFunc.score(model)
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

def replace(pop,popSize,stringLen,scoreFunc):
    if pop.shape[0] == popSize:
        return pop
    score=np.zeros(pop.shape[0])
    for i in range(pop.shape[0]):
        model = get_structure_from_string(pop[i],var)
        score[i] = scoreFunc.score(model)
    newpop = np.zeros([popSize,stringLen])
    for i in range(popSize):
        i1 = np.argmax(score)
        score = np.delete(score,i1)
        newpop[i] = pop[i1]
        pop = np.delete(pop,i1,axis=0)
    return newpop
    
def reGenerate(pop,popSize,var,scoreFunc,stringLen,pc,pm):
    tempop = np.copy(pop)
    for i in range(int(popSize/2)):
        p1,p2,avgScore,popScore = selection(pop,popSize,var,scoreFunc)
        child1,child2 = crossover(pop,p1,p2,pc,stringLen)
        if (len(child1) != 0) and (len(child2) != 0):
            child1 = mutation(child1,pm,stringLen)
            child2 = mutation(child2,pm,stringLen)
            tempop = np.append(tempop,[child1],axis=0)
            tempop = np.append(tempop,[child2],axis=0)
    pop = replace(tempop,popSize,stringLen,scoreFunc)
    return pop


def getStatisticOfGen(pop,popSize,scoreFunc):
    score=np.zeros(pop.shape[0])
    for i in range(pop.shape[0]):
        model = get_structure_from_string(pop[i],var)
        score[i] = scoreFunc.score(model)      
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
        pop = reGenerate(pop,popSize,var,scoreFunc,stringLen,pc,pm)
        popScore,maxScore,avgScore = getStatisticOfGen(pop,popSize,scoreFunc)
        scoresAvg[i+1] += avgScore
        scoresMax[i+1] += maxScore
    q.put(scoresAvg)  
        
if __name__ == '__main__':
    networkName = 'cancer'
    bnNetworkFileName = networkName + '.bif'
    trueStructure = readGroundTruth(networkName)
    bn = BIFReader(path=bnNetworkFileName)
    var = bn.get_variables()
    bnModel = bn.get_model()
    inference = BayesianModelSampling(bnModel)
    sampleSize = 3000
    dataSet = inference.forward_sample(size=sampleSize, return_type='dataframe')
    
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
    #k2=bic
#    print(k2.score(bnModel))
#    print(k2.score(model))
#    
#    print(bic.score(bnModel))
#    print(bic.score(model))
#    
#    print(bdeu.score(bnModel))
#    print(bdeu.score(model))
    
    noOfvar = len(var)
    popSize = 10
    noOfgeneration = 200
    pm = 0.01
    pc = 0.9
    stringLen = int(noOfvar*(noOfvar-1)/2)
    noOfrun = 10
    startTime = time.time()
    #create a Queue to share results
    q = Queue()
    p1 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p1.start()
    p2 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p2.start()
    p3 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p3.start()
    p4 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p4.start()
    p5 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p5.start()
    p6 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p6.start()
    p7 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p7.start()
    p8 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p8.start()
    p9 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p9.start()
    p10 = Process(target=runGA, args=(noOfgeneration,popSize,stringLen,var,pc,pm,k2,q))
    p10.start()
    #runGA(noOfgeneration,popSize,stringLen,var,pc,pm,scoreFunc,q)
    scoresAvg = np.zeros(noOfgeneration)
    scoresMax = np.zeros(noOfgeneration)
    for i in range(noOfrun):
        scoresAvg += q.get(True)
        
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    
    endTime = time.time()
    workTime =  endTime - startTime
    print("The job took " + str(workTime) + " seconds to complete")
#    for run in range(noOfrun):
#        pop = generate_pop(popSize,stringLen)
#        popScore,maxScore,avgScore = getStatisticOfGen(pop,popSize,k2)
#        scoresAvg[0] += avgScore
#        scoresMax[0] += maxScore
#    
#        for i in range(noOfgeneration-1):
#            pop = reGenerate(pop,popSize,var,k2,stringLen,pc,pm)
#            popScore,maxScore,avgScore = getStatisticOfGen(pop,popSize,k2)
#            scoresAvg[i+1] += avgScore
#            scoresMax[i+1] += maxScore
    scoresAvg /= noOfrun
    #model = get_structure_from_string(pop[1],var)
    
    #print(k2.score(model))
    #model.fit(dataSet, estimator=MaximumLikelihoodEstimator)
    #print(k2.score(model))
    #crossover(pop,1,2,0.6,stringLen)
#    scores = np.zeros(noOfgeneration)
#    bestchildll = np.zeros(noOfgeneration)
#    for i in range(noOfgeneration):
#        p1,p2,avgScore,popScore = selection(pop,popSize,var,k2)
#        bestchi = pop[np.argmax(popScore)]
#        bestchildll[i] = getLoglikelihood(list(bestchi), var, dataSet)
#        scores[i] = avgScore
#        child1,child2 = crossover(pop,p1,p2,pc,stringLen)
#        if (len(child1) != 0) and (len(child2) != 0):
#            pop = replace(pop,child1,child2,popScore)
#        pop = mutation(pop,pm,popSize,stringLen)
#        print("generationNo.: ",i)
    #print(p1,p2,avgScore)
    plt.plot(scoresAvg)
    plt.show()
#    plt.plot(bestchildll)   
#    plt.show()
#    
#    plotGraph(trueStructure,var,"trueStructure.png")
#    
#    bStruc,bScore = getBestStructure(pop,popSize,var,k2)
#    plotGraph(bStruc,var,"foundBestStructure.png")
#    print('log*:',getLoglikelihood(list(bStruc), var, dataSet)) 
#    print('logo:',getLoglikelihood(trueStructure, var, dataSet))     
#    print('k2*:', bScore)
#    print('k2o:',k2.score(bnModel))
       
    
    
    #est = HillClimbSearch(dataSet, scoring_method=k2)
    #best_model = est.estimate()
    #print(best_model.edges())
    
    #dataInDict = dataSet.to_dict('index')
    
    #estimator = BayesianEstimator(bnModel, dataSet)
    
    #inference = VariableElimination(bnModel)
    #evidence = dataInDict[0]
    #print(evidence)
    #parameters = estimator.get_parameters()
    #jp = getjointProbability(parameters, evidence,var)
    
    #print('jp:',jp)
    #(equivalent_sample_size=sampleSize)

    #q = inference.query(variables=var)

#    var=['State','Roll','Roll speed','Pitch','pitch speed','Yaw','Yaw speed','Xacc','Yacc','Zacc']
#    st =np.array([1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])    
#    plotGraph(st,var,"trueStructure.png")