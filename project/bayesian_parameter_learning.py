# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:32:07 2018

@author: mrinmoy
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import math
import random


def addMissingData(datatable, variables, isContinuous):
    noOfdiscreteVar = 0
    newVariables = []
    for i in range(len(isContinuous)):
        if not isContinuous[i]:
            if noOfdiscreteVar == 0:
                d = datatable[variables[i]]
                newVariables.append(variables[i])
            else:
                d_temp = datatable[variables[i]]
                d = pd.concat([d,d_temp],axis=1)
                newVariables.append(variables[i])
            noOfdiscreteVar += 1
    allexperiment = [0]*noOfdiscreteVar
    indx1 = 0
    d = pd.DataFrame(d)
    allpossiblecomb = []
    addMissingExperimentData(d,newVariables,allexperiment,indx1,allpossiblecomb)
    colname={}
    for i in range(len(variables)):
        colname[i]=variables[i]
    dNew = pd.DataFrame(allpossiblecomb)
    dNew = dNew.rename(columns=colname)
    if noOfdiscreteVar==len(variables):
        datatable = datatable.append(dNew)
        datatable = datatable.reset_index(drop=True)
    else:
        rtree = DecisionTreeRegressor()
        X = d.as_matrix()
        while noOfdiscreteVar != len(variables):
            y = datatable[variables[noOfdiscreteVar]]
            y=y.as_matrix()
            rtree.fit(X,y)
            y_das = rtree.predict(allpossiblecomb)
            d1 = pd.DataFrame({variables[noOfdiscreteVar]:y_das})
            dNew = pd.concat([dNew, d1],axis=1)
            noOfdiscreteVar += 1
        datatable = datatable.append(dNew)
        datatable = datatable.reset_index(drop=True)
    return datatable

def addMissingExperimentData(datatable,variables,allexperiment,indx,allpossiblecomb):
    discretedata = datatable[variables[0]]
    discretedata = discretedata.unique()
    for i in range(len(discretedata)):
        newvariables = list(variables)
        state = discretedata[i]
        allexperiment[indx] = state
        if len(newvariables)>1:
            del newvariables[0]
            addMissingExperimentData(datatable,newvariables,allexperiment,indx+1,allpossiblecomb)
        else:
            #print(allexperiment)
            allpossiblecomb.append(allexperiment[:])
            #print(allpossiblecomb)
            #print("vvvvvvvv")
            #datatable.loc[len(datatable)] = allexperiment
            #print(len(datatable))

        

def getParam(datatable,variables,isContinuous):
    param={}
    #print(isContinuous)
    if isContinuous[0]:
        if len(variables) > 1 and isContinuous[1]==False:
            datatable = datatable.sort_values(variables[1])
            discretedata = datatable[variables[1]]
            discretedata = discretedata.unique()
            for i in range(len(discretedata)):
                newdatatable = datatable[datatable[variables[1]]==discretedata[i]]
                newdatatable = newdatatable.drop(variables[1],axis=1)
                newvariables = list(variables)
                state = discretedata[i]
                #print('states:')
                #print(state)
                newisContinuous = list(isContinuous)
                discretevar = False
                if len(newisContinuous)>=3 and (not newisContinuous[2]):
                    discretevar = newvariables[2]
                del newvariables[1]
                del newisContinuous[1]
                if not discretevar:
                    newdatatable = newdatatable.reset_index(drop=True)
                    param[state] = getParam(newdatatable,newvariables,newisContinuous)
                else:
                    newdatatable = newdatatable.reset_index(drop=True)
                    param[state] = {discretevar:getParam(newdatatable,newvariables,newisContinuous)}
        else:
            if len(variables) > 1:
                #print("data$$$$$$$$$$$")
                #print(datatable)
                mu = np.array(datatable.mean()) 
                mu1 = mu[0]
                mu2 = mu[1:]
                var = np.array(datatable.cov())
                var11 = var[0][0]
                var12 = var[0][1:]
                var21 = np.transpose(var12)
                var22 = var[1:,1:]  
                #print(var22)
                var22_inv = np.linalg.pinv(var22)
                var2 = np.matmul(var12,var22_inv)
                var = var11-np.matmul(var2,var21)
                param = {'mu1':[mu1],'mu2':mu2,'var':[var],'var2':var2}
                return param
            else:
                mu = np.array(datatable.mean()) 
                var = np.array(datatable.cov())
                param = {'mu':mu,'var':var}
                return param
    else:
        if len(variables)==1:
            discretedata = datatable[variables[0]]
            discretedata = discretedata.unique()
            for i in range(len(discretedata)):
                state = discretedata[i]
                newdatatable = datatable[datatable[variables[0]]==discretedata[i]]
                #print(len(newdatatable))
                param[state] = float(len(newdatatable))/float(len(datatable))
        else:
            datatable = datatable.sort_values(variables[1])
            discretedata = datatable[variables[1]]
            discretedata = discretedata.unique()
            for i in range(len(discretedata)):
                newdatatable = datatable[datatable[variables[1]]==discretedata[i]]
                newdatatable = newdatatable.drop(variables[1],axis=1)
                newvariables = list(variables)
                state = discretedata[i]
                newisContinuous = list(isContinuous)
                discretevar = False
                if len(newisContinuous)>=3 and (not newisContinuous[2]):
                    discretevar = newvariables[2]
                del newvariables[1]
                del newisContinuous[1]
                if not discretevar:
                    newdatatable = newdatatable.reset_index(drop=True)
                    param[state] = getParam(newdatatable,newvariables,newisContinuous)
                else:
                    newdatatable = newdatatable.reset_index(drop=True)
                    param[state] = {discretevar:getParam(newdatatable,newvariables,newisContinuous)}
    return param
    
def getParameters(structure,dataset,variables,isContinuous):
    indx = 0
    parameters = []
    for i in range(len(variables)):
        newVariables = []
        newisContinuous = []
        newisContinuous.append(isContinuous[i])
        newVariables.append(variables[i])       
        d = dataset[variables[i]]
        hasParent = False
        for j in range(i):
            if structure[indx]==1:
                newisContinuous.append(isContinuous[j])
                newVariables.append(variables[j]) 
                d_temp = dataset[variables[j]]
                d = pd.concat([d,d_temp],axis=1)
                hasParent = True
            indx = indx+1
        d = pd.DataFrame(d)
        #print("root data")
        #print(d)
        # if newisContinuous[0] == False and len(newVariables)>1:
        #     allexperiment = [0]*len(variables)
        #     indx1 = 0
            #addMissingExperimentData(d,newVariables,allexperiment,indx1)
        param={} 
        #print(d)
        #print(newisContinuous)
        if hasParent:
            if newisContinuous[1]:
                d = d.reset_index(drop=True)
                param = getParam(d,newVariables,newisContinuous)
            else:
                d = d.reset_index(drop=True)
                param[newVariables[1]] = getParam(d,newVariables,newisContinuous)
        else:
            d = d.reset_index(drop=True)
            param = getParam(d,newVariables,newisContinuous)
        parameters.append(param)
    return parameters
  
def getFactor(variables,parents,parameter,isContinuous,sample,varname,variscont,variableindex):
    if (not parents) or isContinuous[0]:
        #print(isContinuous)
        #print(parameter)
        if variscont:
            if not parents:
                x = sample[variableindex]
                mu = parameter['mu']
                mu = mu[0]
                var = parameter['var']
                var = var[0][0]
                #print('x mu var',x,mu,var)
                factor = getProbabilityForGaussiandist(x,mu,var)
                return factor
            else:
                x = sample[variableindex]
                par = []
                for i in range(len(parents)):
                    temp = sample[variables.index(parents[i])]
                    par.append(temp)
                a = np.array(par)
                mu1 = parameter['mu1']
                mu1 = mu1[0]
                mu2 = parameter['mu2']
                mu2 = mu2[0]
                var2 = parameter['var2']
                mu = mu1 + np.matmul(var2,(a-mu2))
                var = parameter['var']
                var = var[0]
                #print('x mu var',x,mu,var)
                factor = getProbabilityForGaussiandist(x,mu,var)
                return factor
        else:
            temp = (sample[variableindex])
            factor = parameter[temp]
            return factor
    else:
        prnt = parents[0]
        temp = sample[variables.index(prnt)]
        del parents[0]
        del isContinuous[0]
        parameter = parameter[prnt]
        parameter = parameter[temp]
        return getFactor(variables,parents,parameter,isContinuous,sample,varname,variscont,variableindex)

    
def getProbabilityForGaussiandist(x,mu,var):
    A = 1.0/(np.sqrt(2*np.pi*var))
    return A*np.exp(-((x-mu)**2)/(2*var))

def getJointprobability(variables, param, structure, isContinuous,samples):
    #print(param)
    prob = []
    for k in range(samples.shape[0]):
        p = 1.0;
        for i in range(len(variables)):
            parents=[]
            isconti=[]
            for j in range(i):
                if structure[j+int((i*(i-1))/2)]:
                    parents.append(variables[j])
                    isconti.append(isContinuous[j])
            sample = samples.iloc[k,:] #pd.DataFrame({"x1":[0],"x2":[1],"x3":[0],"x4":[1]});
            #print(parents)
            factor = getFactor(variables,parents,param[i],isconti,sample,variables[i],isContinuous[i],i)
            #print('factor',factor)
            if not math.isnan(float(str(factor))):
                p *= factor
        #print(p)
        prob.append(p)
    return prob

def predict(noofclass,variables, param, structure, isContinuous,samples):
    output = []
    for i in range(samples.shape[0]):
        p=[]
        for j in range(noofclass):
            sample = pd.DataFrame({variables[0]:[j]})
            for l in range(len(variables)-1):
                temp = pd.DataFrame({variables[l+1]:[samples[variables[l+1]][i]]})
                sample = pd.concat([sample,temp],axis=1)
            #print(sample)
            prob = getJointprobability(variables, param, structure, isContinuous,sample)
            if prob:
                p.append(prob[0])
        output.append(p.index(max(p)))
    return output

def getSize(variables,structure,isContinuous,samples):
    s = 0
    for i in range(len(variables)):
        hasParent = False
        hasContinusParent = False
        size = 1
        for j in range(i):
            if structure[j+int((i*(i-1))/2)]:
                hasParent = True
                if isContinuous[j]:
                    hasContinusParent = True
                    break
                else:
                    data = samples[variables[j]]
                    data = data.unique()
                    size *= len(data)
        if isContinuous[i]:
            if hasParent:
                if hasContinusParent:
                    size *= 4
                else:
                    size *= 2
            else:
                size *= 2
        else:
            data = samples[variables[i]]
            data = data.unique()
            size *= (len(data)-1)
        s += size
    return s


def getScore(variables, param, structure, isContinuous,samples):
    score = 0
    N = samples.shape[0]
    size = getSize(variables,structure,isContinuous,samples)
    #print('size',size)
    for i in range(N):
        p=[]
        sample = pd.DataFrame({variables[0]:[samples[variables[0]][i]]})
        for l in range(len(variables)-1):
            temp = pd.DataFrame({variables[l+1]:[samples[variables[l+1]][i]]})
            sample = pd.concat([sample,temp],axis=1)
            #print(sample)
        prob = getJointprobability(variables, param, structure, isContinuous,sample)
        #print(prob)
        if prob:
            score += np.log2(prob[0])
    score  = score - (size/2.0)*np.log2(N)
    return score

def getbatchfactor(variables, param, structure, isContinuous,samples,q):
    score = 0
    N = samples.shape[0]
    for i in range(N):
        p=[]
        sample = pd.DataFrame({variables[0]:[samples[variables[0]][i]]})
        for l in range(len(variables)-1):
            temp = pd.DataFrame({variables[l+1]:[samples[variables[l+1]][i]]})
            sample = pd.concat([sample,temp],axis=1)
            #print(sample)
        prob = getJointprobability(variables, param, structure, isContinuous,sample)
        #print(prob)
        if prob:
            score += np.log2(prob[0])
    q.put(score)