# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:32:07 2018

@author: mrinmoy
"""

import pandas as pd
import numpy as np


def addMissingExperimentData(datatable,variables,allexperiment,indx):
    discretedata = datatable[variables[0]]
    discretedata = discretedata.unique()
    for i in range(len(discretedata)):
        newvariables = list(variables)
        state = discretedata[i]
        allexperiment[indx] = state
        if len(newvariables)>1:
            del newvariables[0]
            addMissingExperimentData(datatable,newvariables,allexperiment,indx+1)
        else:
            #print(allexperiment)
            datatable.loc[len(datatable)] = allexperiment
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
                del newvariables[1]
                newisContinuous = list(isContinuous)
                del newisContinuous[1]
                param[state] = getParam(newdatatable,newvariables,newisContinuous)
        else:
            if len(variables) > 1:
                mu = np.array(datatable.mean()) 
                mu1 = mu[0]
                mu2 = mu[1:]
                var = np.array(datatable.cov())
                var11 = var[0][0]
                var12 = var[0][1:]
                var21 = np.transpose(var12)
                var22 = var[1:,1:]  
                var22_inv = np.linalg.pinv(var22)
                var = var11-np.matmul(np.matmul(var12,var22_inv),var21)
                param = {'mu1':mu1,'mu2':mu2,'var':var,'var12':var12,'var22_inv':var22_inv}
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
                del newvariables[1]
                newisContinuous = list(isContinuous)
                del newisContinuous[1]
                param[state] = getParam(newdatatable,newvariables,newisContinuous)
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
        if newisContinuous[0] == False and len(newVariables)>1:
            allexperiment = [0]*len(variables)
            indx1 = 0
            addMissingExperimentData(d,newVariables,allexperiment,indx1)
        param={} 
        #print(d)
        #print(newisContinuous)
        if hasParent:
            param[newVariables[1]] = getParam(d,newVariables,newisContinuous)
        else:
            param = getParam(d,newVariables,newisContinuous)
        parameters.append(param)
    return parameters
    
    
    
if __name__ == '__main__':
#    dataset = pd.DataFrame({"x1":[7,7,8,8,8],"x2":[3,6,6,6,3],"x3":[5,6,6,6,5]})
#    variables = ['x1','x2','x3']
#    isContinuous=[False,False,False]
#    param={}   
##    allexperiment = [0]*len(variables)
##    indx = 0
##    addMissingExperimentData(dataset,variables,allexperiment,indx)
#    param[variables[1]] = getParam(dataset,variables,isContinuous)
    structure = [1,1,0]
    dataset = pd.DataFrame({"x1":[0,0,1],"x2":[3,3,6],"x3":[5,6,10]})
    variables = ['x1','x2','x3']
    isContinuous=[False,True,True]
    parm = getParameters(structure,dataset,variables,isContinuous)