from sklearn import datasets
import random
import pandas as pd





if __name__ == '__main__':
	# breast_cancer = datasets.load_breast_cancer()
	# #wine = datasets.load_wine()
	# X = breast_cancer.data  
	# y = breast_cancer.target
	# noOffeature = X.shape[1]
 # 	variables = []
 # 	variables.append('y')
 # 	indx = [i for i in range(len(y))]
	# random.shuffle(indx)
	# totaltrainsample = int(0.8*len(y))
	# dataset = pd.DataFrame({"y":y[indx[0:totaltrainsample]]})
 # 	for i in range(noOffeature):
 # 		var = 'x'+str(i+1)
 # 		variables.append(var)
 # 		temp = pd.DataFrame({var:X[indx[0:totaltrainsample],i]})
 # 		dataset = pd.concat([dataset,temp],axis=1)
 # 		#dataset = dataset.reset_index(drop=True)
 # 	testset = pd.DataFrame({"y":y[indx[totaltrainsample:len(y)]]})
 # 	for i in range(noOffeature):
 # 		var = 'x'+str(i+1)
 # 		temp = pd.DataFrame({var:X[indx[totaltrainsample:len(y)],i]})
 # 		testset = pd.concat([testset,temp],axis=1)
 # 	ytrue = y[indx[totaltrainsample:len(y)]]
 # 	print(testset)
 # 	print(variables)
 # 	print(dataset)
 # 	print(ytrue)
    train_data_df = pd.read_json('dataset.json')
    dataset = train_data_df.reset_index(drop=True)
    state = dataset['state']
    state = state.replace('Hold',0)
    state = state.replace('Fly Orbit and Observe',1)
    state = state.replace('Fly Search Pattern',2)
    state = state.replace('Survey Target',3)
    print(state[0])
    dataset = pd.DataFrame({'y':state[0:10],'x1':dataset['roll'][0:10],'x2':dataset['pitch'][0:10],'x3':dataset['yaw'][0:10],\
    	'x4':dataset['rollspeed'][0:10],'x5':dataset['pitchspeed'][0:10],'x6':dataset['yawspeed'][0:10],'x7':dataset['xacc'][0:10],\
    	'x8':dataset['yacc'][0:10],'x9':dataset['zacc'][0:10]})
    print(dataset)
 	
