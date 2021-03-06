import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

def PlotRes(data, path):
    """ 
        data - pd dataframe 'history' from train
    """

    MSE_xGT = data.iloc[:, 1]  # MSE
    MSE_xGT = MSE_xGT[1:]
    MSE_xGT = MSE_xGT.to_numpy()
    MSE_xGT = MSE_xGT.astype(float)

    MSE_AVG = data.iloc[:, 6]
    MSE_AVG = MSE_AVG[1:]
    MSE_AVG = MSE_AVG.to_numpy()
    MSE_AVG = MSE_AVG.astype(float)

    plt.plot(np.arange(len(MSE_xGT)),MSE_xGT, label = 'MSE: x-x_GT')
    plt.plot(np.arange(len(MSE_AVG)),MSE_AVG, label = 'MSE: Avgeraging')
    plt.legend()

    image_path = path + '/ResultGraph.png'
    logging.info("\n writing graph to " + image_path)
    plt.savefig(image_path)
    plt.close()

def PlotAll(Dict, path): 
    for key in Dict: 
        plt.plot(np.arange(len(Dict[key])), Dict[key], label = key) 
    
    plt.legend()
    image_path = path + '/AllGraphs.png'
    logging.info("\n writing graph to " + image_path)
    plt.savefig(image_path)
    plt.close() 

    df = pd.DataFrame.from_dict(Dict)
    df.to_csv(path + '/AllMSE.csv')