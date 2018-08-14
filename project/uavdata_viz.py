import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import load_dataset as lddata



if __name__ == '__main__':
    noofclass,variables,isContinuous,dataset,testset,ytrue = lddata.load_uav_state_data_without_randomized()
    dataset = dataset.sort_values('y')
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.dropna()
    print(dataset)
    x1 = dataset['x1']
    x2 = dataset['x2']
    x = x1**2 + x2**2
    df2 = pd.DataFrame({'x10':x})
    
    df3 = pd.concat([dataset,df2], axis=1)
    print(df3)
    y = dataset['y']
    plt.subplot(211)
    plt.plot(y)
    plt.subplot(212)
    plt.plot(x)
    plt.show()
