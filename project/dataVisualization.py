from sklearn import datasets
import random
import pandas as pd





if __name__ == '__main__':
	breast_cancer = datasets.load_breast_cancer()
	#wine = datasets.load_wine()
	X = breast_cancer.data  
	y = breast_cancer.target
	noOffeature = X.shape[1]
 	variables = []
 	variables.append('y')
 	indx = [i for i in range(len(y))]
	random.shuffle(indx)
	totaltrainsample = int(0.8*len(y))
	dataset = pd.DataFrame({"y":y[indx[0:totaltrainsample]]})
 	for i in range(noOffeature):
 		var = 'x'+str(i+1)
 		variables.append(var)
 		temp = pd.DataFrame({var:X[indx[0:totaltrainsample],i]})
 		dataset = pd.concat([dataset,temp],axis=1)
 		#dataset = dataset.reset_index(drop=True)
 	testset = pd.DataFrame({"y":y[indx[totaltrainsample:len(y)]]})
 	for i in range(noOffeature):
 		var = 'x'+str(i+1)
 		temp = pd.DataFrame({var:X[indx[totaltrainsample:len(y)],i]})
 		testset = pd.concat([testset,temp],axis=1)
 	ytrue = y[indx[totaltrainsample:len(y)]]
 	print(testset)
 	print(variables)
 	print(dataset)
 	print(ytrue)
 	
