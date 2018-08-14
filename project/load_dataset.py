from sklearn import datasets
import random
import pandas as pd
import numpy as np
import time
import sys



def load_iris_data():
	iris = datasets.load_iris()
	X = iris.data  
	y = iris.target
	noofclass = 3
	variables = ['y','x1','x2','x3','x4']
	isContinuous=[False,True,True,True,True]
	indx = [i for i in range(len(y))]
	random.shuffle(indx)
	totaltrainsample = int(0.8*len(y))
 	dataset = pd.DataFrame({"y":y[indx[0:totaltrainsample]],"x1":X[indx[0:totaltrainsample],0],"x2":X[indx[0:totaltrainsample],1],"x3":X[indx[0:totaltrainsample],2],"x4":X[indx[0:totaltrainsample],3]})
	testset = pd.DataFrame({"y":y[indx[totaltrainsample:len(y)]],"x1":X[indx[totaltrainsample:len(y)],0],"x2":X[indx[totaltrainsample:len(y)],1],"x3":X[indx[totaltrainsample:len(y)],2],"x4":X[indx[totaltrainsample:len(y)],3]})
	ytrue = y[indx[totaltrainsample:len(y)]]
	return noofclass,variables,isContinuous,dataset,testset,ytrue

def load_breast_data():
	breast_cancer = datasets.load_breast_cancer()
	#wine = datasets.load_wine()
	X = breast_cancer.data  
	y = breast_cancer.target
	noOffeature = X.shape[1]
	noofclass = 2
 	variables = []
 	isContinuous = []
 	variables.append('y')
 	isContinuous.append(False)
 	indx = [i for i in range(len(y))]
	random.shuffle(indx)
	totaltrainsample = int(0.8*len(y))
	dataset = pd.DataFrame({"y":y[indx[0:totaltrainsample]]})
 	for i in range(noOffeature):
 		var = 'x'+str(i+1)
 		variables.append(var)
 		isContinuous.append(True)
 		temp = pd.DataFrame({var:X[indx[0:totaltrainsample],i]})
 		dataset = pd.concat([dataset,temp],axis=1)
 		#dataset = dataset.reset_index(drop=True)
 	testset = pd.DataFrame({"y":y[indx[totaltrainsample:len(y)]]})
 	for i in range(noOffeature):
 		var = 'x'+str(i+1)
 		temp = pd.DataFrame({var:X[indx[totaltrainsample:len(y)],i]})
 		testset = pd.concat([testset,temp],axis=1)
 	ytrue = y[indx[totaltrainsample:len(y)]]
 	return noofclass,variables,isContinuous,dataset,testset,ytrue

def load_uav_state_data():
	train_data_df = pd.read_json('dataset.json')
	dataset = train_data_df.reset_index(drop=True)
	dataset = dataset.dropna()
	dataset['roll'] = dataset['roll'].apply(lambda x: x * 60)
	dataset['pitch'] = dataset['pitch'].apply(lambda x: x * 60)
	dataset['yaw'] = dataset['yaw'].apply(lambda x: x * 60)
	dataset['rollspeed'] = dataset['rollspeed'].apply(lambda x: x * 60)
	dataset['pitchspeed'] = dataset['pitchspeed'].apply(lambda x: x * 60)
	dataset['yawspeed'] = dataset['yawspeed'].apply(lambda x: x * 60)
	dataset = dataset.reset_index(drop=True)
	dataset = dataset.dropna()
	state = dataset['state']
	state = state.replace('Hold',0)
	state = state.replace('Fly Orbit and Observe',1)
	state = state.replace('Fly Search Pattern',2)
	state = state.replace('Survey Target',3)
	state = state.astype(int)
	y = state
	noOffeature = 9
	noofclass = 4
 	variables = []
 	isContinuous = []
 	variables.append('y')
 	isContinuous.append(False)
 	indx = [i for i in range(len(y))]
	random.shuffle(indx)
	totaltrainsample = int(0.8*len(y))
	trainset = pd.DataFrame({'y':state[indx[0:totaltrainsample]],'x1':dataset['roll'][indx[0:totaltrainsample]],\
    	'x2':dataset['pitch'][indx[0:totaltrainsample]],'x3':dataset['yaw'][indx[0:totaltrainsample]],\
    	'x4':dataset['rollspeed'][indx[0:totaltrainsample]],'x5':dataset['pitchspeed'][indx[0:totaltrainsample]],\
    	'x6':dataset['yawspeed'][indx[0:totaltrainsample]],'x7':dataset['xacc'][indx[0:totaltrainsample]],\
    	'x8':dataset['yacc'][indx[0:totaltrainsample]],'x9':dataset['zacc'][indx[0:totaltrainsample]]})
	trainset = trainset.reset_index(drop=True)
	trainset = trainset.dropna()
 	for i in range(noOffeature):
 		var = 'x'+str(i+1)
 		variables.append(var)
 		isContinuous.append(True)
 	testset = pd.DataFrame({'y':state[indx[totaltrainsample:len(y)]],'x1':dataset['roll'][indx[totaltrainsample:len(y)]],\
    	'x2':dataset['pitch'][indx[totaltrainsample:len(y)]],'x3':dataset['yaw'][indx[totaltrainsample:len(y)]],\
    	'x4':dataset['rollspeed'][indx[totaltrainsample:len(y)]],'x5':dataset['pitchspeed'][indx[totaltrainsample:len(y)]],\
    	'x6':dataset['yawspeed'][indx[totaltrainsample:len(y)]],'x7':dataset['xacc'][indx[totaltrainsample:len(y)]],\
    	'x8':dataset['yacc'][indx[totaltrainsample:len(y)]],'x9':dataset['zacc'][indx[totaltrainsample:len(y)]]})
 	testset = testset.reset_index(drop=True)
 	testset = testset.dropna()
 	#print(y)
 	ytrue = list(y[indx[totaltrainsample:len(y)]])
 	return noofclass,variables,isContinuous,trainset,testset,ytrue


def load_uav_state_data_without_randomized():
	train_data_df = pd.read_json('dataset.json')
	dataset = train_data_df.reset_index(drop=True)
	dataset = dataset.dropna()
	dataset['roll'] = dataset['roll'].apply(lambda x: x * 60)
	dataset['pitch'] = dataset['pitch'].apply(lambda x: x * 60)
	dataset['yaw'] = dataset['yaw'].apply(lambda x: x * 60)
	dataset['rollspeed'] = dataset['rollspeed'].apply(lambda x: x * 60)
	dataset['pitchspeed'] = dataset['pitchspeed'].apply(lambda x: x * 60)
	dataset['yawspeed'] = dataset['yawspeed'].apply(lambda x: x * 60)
	dataset = dataset.reset_index(drop=True)
	dataset = dataset.dropna()
	state = dataset['state']
	state = state.replace('Hold',0)
	state = state.replace('Fly Orbit and Observe',1)
	state = state.replace('Fly Search Pattern',2)
	state = state.replace('Survey Target',3)
	state = state.astype(int)
	y = state
	noOffeature = 15
	noofclass = 4
 	variables = []
 	isContinuous = []
 	variables.append('y')
 	isContinuous.append(False)
 	indx = [i for i in range(len(y))]
	# random.shuffle(indx)
	totaltrainsample = int(0.8*len(y))
	trainset = pd.DataFrame({'y':state[indx[0:totaltrainsample]],'x1':dataset['roll'][indx[0:totaltrainsample]],\
    	'x2':dataset['pitch'][indx[0:totaltrainsample]],'x3':dataset['yaw'][indx[0:totaltrainsample]],\
    	'x4':dataset['rollspeed'][indx[0:totaltrainsample]],'x5':dataset['pitchspeed'][indx[0:totaltrainsample]],\
    	'x6':dataset['yawspeed'][indx[0:totaltrainsample]],'x7':dataset['xacc'][indx[0:totaltrainsample]],\
    	'x8':dataset['yacc'][indx[0:totaltrainsample]],'x9':dataset['zacc'][indx[0:totaltrainsample]]})
	trainset = trainset.reset_index(drop=True)
	trainset = trainset.dropna()
	x1 = trainset['x1']
	x2 = trainset['x2']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x10':x})
	trainset = pd.concat([trainset,df2], axis=1)
	trainset = trainset.reset_index(drop=True)
	trainset = trainset.dropna()
	x1 = trainset['x2']
	x2 = trainset['x3']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x11':x})
	trainset = pd.concat([trainset,df2], axis=1)
	trainset = trainset.reset_index(drop=True)
	trainset = trainset.dropna()
	x1 = trainset['x1']
	x2 = trainset['x3']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x12':x})
	trainset = pd.concat([trainset,df2], axis=1)
	trainset = trainset.reset_index(drop=True)
	trainset = trainset.dropna()
	x1 = trainset['x4']
	x2 = trainset['x5']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x13':x})
	trainset = pd.concat([trainset,df2], axis=1)
	trainset = trainset.reset_index(drop=True)
	trainset = trainset.dropna()
	x1 = trainset['x5']
	x2 = trainset['x6']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x14':x})
	trainset = pd.concat([trainset,df2], axis=1)
	trainset = trainset.reset_index(drop=True)
	trainset = trainset.dropna()
	x1 = trainset['x4']
	x2 = trainset['x6']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x15':x})
	trainset = pd.concat([trainset,df2], axis=1)
	trainset = trainset.reset_index(drop=True)
	trainset = trainset.dropna()

 	for i in range(noOffeature):
 		var = 'x'+str(i+1)
 		variables.append(var)
 		isContinuous.append(True)
 	testset = pd.DataFrame({'y':state[indx[totaltrainsample:len(y)]],'x1':dataset['roll'][indx[totaltrainsample:len(y)]],\
    	'x2':dataset['pitch'][indx[totaltrainsample:len(y)]],'x3':dataset['yaw'][indx[totaltrainsample:len(y)]],\
    	'x4':dataset['rollspeed'][indx[totaltrainsample:len(y)]],'x5':dataset['pitchspeed'][indx[totaltrainsample:len(y)]],\
    	'x6':dataset['yawspeed'][indx[totaltrainsample:len(y)]],'x7':dataset['xacc'][indx[totaltrainsample:len(y)]],\
    	'x8':dataset['yacc'][indx[totaltrainsample:len(y)]],'x9':dataset['zacc'][indx[totaltrainsample:len(y)]]})
 	testset = testset.reset_index(drop=True)
 	testset = testset.dropna()
 	x1 = testset['x1']
	x2 = testset['x2']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x10':x})
	testset = pd.concat([testset,df2], axis=1)
	testset = testset.reset_index(drop=True)
	testset = testset.dropna()
 	x1 = testset['x2']
	x2 = testset['x3']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x11':x})
	testset = pd.concat([testset,df2], axis=1)
	testset = testset.reset_index(drop=True)
	testset = testset.dropna()
 	x1 = testset['x1']
	x2 = testset['x3']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x12':x})
	testset = pd.concat([testset,df2], axis=1)
	testset = testset.reset_index(drop=True)
	testset = testset.dropna()
	x1 = testset['x4']
	x2 = testset['x5']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x13':x})
	testset = pd.concat([testset,df2], axis=1)
	testset = testset.reset_index(drop=True)
	testset = testset.dropna()
	x1 = testset['x5']
	x2 = testset['x6']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x14':x})
	testset = pd.concat([testset,df2], axis=1)
	testset = testset.reset_index(drop=True)
	testset = testset.dropna()
	x1 = testset['x4']
	x2 = testset['x6']
	x = x1**2 + x2**2
	df2 = pd.DataFrame({'x15':x})
	testset = pd.concat([testset,df2], axis=1)
	testset = testset.reset_index(drop=True)
	testset = testset.dropna()
 	#print(y)
 	ytrue = list(y[indx[totaltrainsample:len(y)]])
 	trainset.to_csv(path_or_buf="trainset.csv")
 	testset.to_csv(path_or_buf="testset.csv")
 	return noofclass,variables,isContinuous,trainset,testset,ytrue