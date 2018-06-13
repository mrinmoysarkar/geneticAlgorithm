# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:32:07 2018

@author: mrinmoy
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def addMissingData(datatable, variables, isContinuous):
    noOfdiscreteVar = 0
    newVariables = []
    for i in range(len(isContinuous)):
        if not isContinuous[i]:
            if noOfdiscreteVar == 0:
                d = dataset[variables[i]]
                newVariables.append(variables[i])
            else:
                d_temp = dataset[variables[i]]
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
        print(X)
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
                newisContinuous = list(isContinuous)
                discretevar = False
                if len(newisContinuous)>=3 and (not newisContinuous[2]):
                    discretevar = newvariables[2]
                del newvariables[1]
                del newisContinuous[1]
                if not discretevar:
                    param[state] = getParam(newdatatable,newvariables,newisContinuous)
                else:
                    param[state] = {discretevar:getParam(newdatatable,newvariables,newisContinuous)}
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
                newisContinuous = list(isContinuous)
                discretevar = False
                if len(newisContinuous)>=3 and (not newisContinuous[2]):
                    discretevar = newvariables[2]
                del newvariables[1]
                del newisContinuous[1]
                if not discretevar:
                    param[state] = getParam(newdatatable,newvariables,newisContinuous)
                else:
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
        # if newisContinuous[0] == False and len(newVariables)>1:
        #     allexperiment = [0]*len(variables)
        #     indx1 = 0
            #addMissingExperimentData(d,newVariables,allexperiment,indx1)
        param={} 
        #print(d)
        #print(newisContinuous)
        if hasParent:
            param[newVariables[1]] = getParam(d,newVariables,newisContinuous)
        else:
            param = getParam(d,newVariables,newisContinuous)
        parameters.append(param)
    return parameters
    
    
def getProbabilityForGaussiandist(x,mu,var):
    pass

def getJointprobability(variables, param, structure):
    for i in range(len(variables)):
        for j in range(i):
            if structure[j]:
                pass


if __name__ == '__main__':
#    dataset = pd.DataFrame({"x1":[7,7,8,8,8],"x2":[3,6,6,6,3],"x3":[5,6,6,6,5]})
#    variables = ['x1','x2','x3']
#    isContinuous=[False,False,False]
#    param={}   
##    allexperiment = [0]*len(variables)
##    indx = 0
##    addMissingExperimentData(dataset,variables,allexperiment,indx)
#    param[variables[1]] = getParam(dataset,variables,isContinuous)
    structure = [1,1,1,1,1,1]
    dataset = pd.DataFrame({"x1":[0,0,1],"x2":[1,0,1],"x3":[1,1,0],"x4":[1,1,0]})
    variables = ['x1','x2','x3','x4']
    isContinuous=[False,False,False,True]
    dataset = addMissingData(dataset, variables, isContinuous)
    print(dataset)
    parm = getParameters(structure,dataset,variables,isContinuous)
    print(parm)