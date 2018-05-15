# -*- coding: utf-8 -*-
"""
Created on Mon May 14 09:34:28 2018

@author: mrinmoy
"""

import pandas as pd
import numpy as np


def getParameters(structure,dataset,variables,isContinuous):
    indx = 0
    parameters = []
    for i in range(len(variables)):
        d = dataset[variables[i]]
        hasParent = False
        for j in range(i):
            if structure[indx]==1:
                d_temp = dataset[variables[j]]
                d = pd.concat([d,d_temp],axis=1)
                hasParent = True
            indx = indx+1
        if hasParent:
            mu = np.array(d.mean()) 
            mu1 = mu[0]
            mu2 = mu[1:]
            var = np.array(d.cov())
            var11 = var[0][0]
            var12 = var[0][1:]
            var21 = np.transpose(var12)
            var22 = var[1:,1:]  
            var22_inv = np.linalg.pinv(var22)
            var = var11-np.matmul(np.matmul(var12,var22_inv),var21)
            param = {'mu1':mu1,'mu2':mu2,'var':var,'var12':var12,'var22_inv':var22_inv}
        else:
            mu = np.array(d.mean()) 
            var = np.array(d.var())
            param = {'mu':mu,'var':var}
        parameters.append(param)
    return parameters


if __name__ == '__main__':
    structure = [1,1,0]
    dataset = pd.DataFrame({"x1":[0,0,1],"x2":[3,3,6],"x3":[5,6,10]})
    variables = ['x1','x2','x3']
    isContinuous=[False,True,True]
    parm = getParameters(structure,dataset,variables,isContinuous)
        
            