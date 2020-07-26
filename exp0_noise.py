import time, os
import torch
import scipy.misc
import numpy as np
from cv2 import cv2
import sys
import logging

from plots import PlotRes
import dataset, generator, discriminator, train

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def SweepThru(noiseSigma = 0.5,ns = 100, bs = 10, lr_G = 1e-2, lr_D = 1e-2, ns_G = 1, ns_D = 3, regW = 1e1, path = "dir"):


    t = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(
        'results', path,
        'noise_' + t)

    os.makedirs(outdir)


    # true data distribution
    x_GT = scipy.misc.face()
    x_GT = np.mean(x_GT/255, axis=2)
    x_GT = x_GT[100:100+256, 400:400+256]
    x_GT = x_GT - np.mean(x_GT)

    """
    #------------------------------------------------
    x_GT = cv2.imread("images/dice.jpg",1)
    print(x_GT)

    factor = np.max(np.max(np.array(x_GT.shape) / 256), initial = 1)
    #x_GT = cv2.resize(x_GT,fx = 1/factor, fy=1/factor)
    x_GT = cv2.resize(x_GT, dsize = (int(x_GT.shape[0]//factor), int(x_GT.shape[1]//factor)))


    print(x_GT)
    #x_GT.astype(int)
    # Resize the image using cv2
    x_GT = np.mean(x_GT/255, axis=2)
    #x_GT = x_GT[100:100+256, 400:400+256]
    x_GT = x_GT - np.mean(x_GT)
    #print(x_GT)
    #------------------------------------------------
    """
    forward_model = generator.AddNoise(sigma = noiseSigma)

    G_true = generator.Generator(x0=x_GT, model=forward_model)

    dataset_true = dataset.GeneratorWrapper(G_true)

    # generator
    G = generator.Generator(shape=x_GT.shape, model=forward_model)

    # discriminator
    D = discriminator.ConvMax(x_GT.size)

    logging.info(
                f"\n"
                f"Num Steps: {ns}\n"
                f"Batch Size: {bs}\n"
                f"Learning Rate G: {lr_G}\n"
                f"Learning Rate D: {lr_D}\n"
                f"Num Steps G: {ns_G}\n" 
                f"Num Steps D:  {ns_D}\n"
                f"Reg Weight: {regW}\n"
                f"Noise Sigma: {noiseSigma}\n"
                )



    # training
    G, D, history = train.train_WGAN(D, G, dataset_true,
                            num_steps=ns, batch_size=bs,
                            learning_rate_G=lr_G, learning_rate_D=lr_D,
                            num_steps_G=ns_G, num_steps_D=ns_G,
                            reg_weight=regW, device=device,
                        out_folder=outdir,x_gt = x_GT)

    print(history)

    history.to_csv(outdir + '/history.csv')

    PlotRes(history, outdir)

#noise_sigmas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
num_steps_dis = [1,2,3,4,5,6,7,8,9,10]
#batchSizes = [4]
#num_steps_gen = [1,2,3,4,5,6,7,8,9,10]

t = time.strftime("%Y-%m-%d-%H-%M-%S")
top_dir = "results/" + t + "D_StepTrials"
os.makedirs(top_dir)

# setup logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
log_filename = os.path.join(top_dir, 'log.txt')
logging.basicConfig(  # sending to file
    filename=log_filename,
    level=logging.DEBUG,
    format='%(message)s',
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # sending to stdout



for inc in num_steps_dis:
    SweepThru(ns_D = inc, path = (t+"D_StepTrials"), ns=100)
