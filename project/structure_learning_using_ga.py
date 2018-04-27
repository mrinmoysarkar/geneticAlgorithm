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


if __name__ == '__main__':
    bnNetworkFileName = 'asia.bif'
    bn = BIFReader(path=bnNetworkFileName)
    var = bn.get_variables()
    bnModel = bn.get_model()
    inference = BayesianModelSampling(bnModel)
    dataSet = inference.forward_sample(size=1000, return_type='dataframe')
    
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
    
    print(k2.score(bnModel))
    print(k2.score(model))
    
    print(bic.score(bnModel))
    print(bic.score(model))
    
    print(bdeu.score(bnModel))
    print(bdeu.score(model))