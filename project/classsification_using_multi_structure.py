import bayesian_parameter_learning as bpl
import ga_operations as gaop
from sklearn import datasets
import random
import pandas as pd
import numpy as np
import time
import sys
from multiprocessing import Process, Queue

class scoreFunction:
	
	def __init__(self,datatable,variables,isContinuous):
		self.data = bpl.addMissingData(datatable, variables, isContinuous)
		self.variables = variables
		self.isContinuous = isContinuous
	
	def score(self,structure):
		#print(structure)
		param = bpl.getParameters(structure,self.data,self.variables,self.isContinuous)
		#print(param)
		scor = bpl.getScore(self.variables, param, structure, self.isContinuous,self.data)
		return scor
	
	def calc_param(self,structures):
		n = structures.shape[0]
		self.structures = structures
		self.parameters = []
		for i in range(n):
			param = bpl.getParameters(structures[i],self.data,self.variables,self.isContinuous)
			self.parameters.append(param)
	
	def calc_weight(self):
		n = self.structures.shape[0]
		self.weight = []
		for i in range(n):
			param = self.parameters[i]
			scor = bpl.getScore(self.variables, param, self.structures[i], self.isContinuous,self.data)
			self.weight.append(scor)
		sumw = sum(self.weight)
		self.weight = [x / sumw for x in self.weight]
		print(self.weight)
	
	def predict_multi(self,noofclass,samples):
		output = []
		n = self.structures.shape[0]
		for i in range(samples.shape[0]):
			p = [] 
			for j in range(noofclass):
				sample = pd.DataFrame({self.variables[0]:[j]})
				for l in range(len(self.variables)-1):
					temp = pd.DataFrame({self.variables[l+1]:[samples[self.variables[l+1]][i]]})
					sample = pd.concat([sample,temp],axis=1)
				pro = 0
				for s in range(n):
					w = self.weight[s]
					structure = self.structures[s]
					param = self.parameters[s]
					prob = bpl.getJointprobability(self.variables, param, structure, self.isContinuous,sample)
					if prob:
						pro += w*prob[0]
				p.append(pro)
			output.append(p.index(max(p)))
		return output
	



if __name__ == '__main__':
#    dataset = pd.DataFrame({"x1":[7,7,8,8,8],"x2":[3,6,6,6,3],"x3":[5,6,6,6,5]})
#    variables = ['x1','x2','x3']
#    isContinuous=[False,False,False]
#    param={}   
##    allexperiment = [0]*len(variables)
##    indx = 0
##    addMissingExperimentData(dataset,variables,allexperiment,indx)
#    param[variables[1]] = getParam(dataset,variables,isContinuous)
    # structure = [1,1,1,1,1,1]
    # dataset = pd.DataFrame({"x1":[0,0,1],"x2":[1,0,1],"x3":[1,1,0],"x4":[1,1,0]})
    # variables = ['x1','x2','x3','x4']
    # isContinuous=[False,False,True,True]
    # dataset = addMissingData(dataset, variables, isContinuous)
    # param = getParameters(structure,dataset,variables,isContinuous)
    # getJointprobability(variables, param, structure,isContinuous,dataset)

    iris = datasets.load_iris()
    X = iris.data  
    y = iris.target
    noofclass = 3
    #print(X)
    #print(y)
    indx = [i for i in range(len(y))]
    random.shuffle(indx)
    totaltrainsample = int(0.8*len(y))
    #print(X[indx[0:totaltrainsample],1])
    #print(y[indx[0:totaltrainsample]])
    #print(totaltrainsample)
    #structure = [1,1,0,1,1,0,1,0,1,0]
    dataset = pd.DataFrame({"y":y[indx[0:totaltrainsample]],"x1":X[indx[0:totaltrainsample],0],"x2":X[indx[0:totaltrainsample],1],"x3":X[indx[0:totaltrainsample],2],"x4":X[indx[0:totaltrainsample],3]})
    variables = ['y','x1','x2','x3','x4']
    isContinuous=[False,True,True,True,True]
    scorefunc = scoreFunction(dataset,variables,isContinuous)
    #print(scorefunc.score(structure))

    #dataset = bpl.addMissingData(dataset, variables, isContinuous)
    #param = bpl.getParameters(structure,dataset,variables,isContinuous)
    #print(param)

    testset = pd.DataFrame({"y":y[indx[totaltrainsample:len(y)]],"x1":X[indx[totaltrainsample:len(y)],0],"x2":X[indx[totaltrainsample:len(y)],1],"x3":X[indx[totaltrainsample:len(y)],2],"x4":X[indx[totaltrainsample:len(y)],3]})
    ytrue = y[indx[totaltrainsample:len(y)]]
    #noofclass = 3
    #ypredict = bpl.predict(noofclass,variables, param, structure, isContinuous,testset)
    #print(ypredict-ytrue)
    #print(len(ypredict-ytrue))
    #print(bpl.getScore(variables, param, structure, isContinuous,testset))


    noOfvar = len(variables)
    popSize = 10
    noOfgeneration = 10
    pm = 0.01
    pc = 0.9
    stringLen = int(noOfvar*(noOfvar-1)/2)
    pop = gaop.generate_pop(popSize,stringLen)
    noOfrun = 2
    #startTime = time.time()
    #create a Queue to share results
    q = 0 #Queue()
    structures = gaop.runGA(noOfgeneration,popSize,stringLen,variables,pc,pm,scorefunc,q)
    print(structures)
    scorefunc.calc_param(structures)
    scorefunc.calc_weight()
    ypredict = scorefunc.predict_multi(noofclass,testset)
    diff = list(ypredict-ytrue)
    correct_prediction = diff.count(0)
    print(correct_prediction)
    print(len(ypredict-ytrue))
    pred_correct = (float(correct_prediction)/len(diff))*100.0
    print("correct prediction % :", pred_correct)

    # p1 = Process(target=gaop.runGA, args=(noOfgeneration,popSize,stringLen,variables,pc,pm,scorefunc,q))
    # p1.start()
    # p2 = Process(target=gaop.runGA, args=(noOfgeneration,popSize,stringLen,variables,pc,pm,scorefunc,q))
    # p2.start()
    # scoresAvg = np.zeros(noOfgeneration)
    # scoresMax = np.zeros(noOfgeneration)
    # for i in range(noOfrun):
    #     scoresAvg += q.get(True)
        
    # p1.join()
    # p2.join()
    # endTime = time.time()
    # workTime =  endTime - startTime
    # print("The job took " + str(workTime) + " seconds to complete")
    