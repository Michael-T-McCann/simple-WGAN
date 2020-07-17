import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import sys

def PlotRes(path):
    sys.stdout.close

    data = pd.read_csv(path + '/res.txt', sep="\t", header=None)  
    MSE_xGT = data[1]  
    MSE_xGT = MSE_xGT[1:]
    MSE_xGT = MSE_xGT.to_numpy()
    MSE_xGT = MSE_xGT.astype(float)

    MSE_AVG = data[6]
    MSE_AVG = MSE_AVG[1:]
    MSE_AVG = MSE_AVG.to_numpy()
    MSE_AVG = MSE_AVG.astype(float) 
    MSE_AVG[0] = 0.0 
 
    
    plt.plot(np.arange(100),MSE_xGT, label = 'MSE: x-x_GT')
    plt.plot(np.arange(100),MSE_AVG, label = 'MSE: Avgeraging')  

    plt.savefig(path + '/ResultGraph.png') 


PlotRes("./results/20200716-191431Batch_Trials/noise_20200716-191431")