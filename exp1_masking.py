import time, os
import torch
import scipy.misc
import numpy as np

import dataset, generator, discriminator, train

# setup output directory
outdir = os.path.join(
    'results',
    'masking_' + time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(outdir)


# true data distribution
x_GT = scipy.misc.face()
x_GT = np.mean(x_GT/255, axis=2)
x_GT = x_GT[100:100+256, 400:400+256]
x_GT = x_GT - np.mean(x_GT)

forward_model = torch.nn.Sequential(
    generator.AddNoise(sigma=0.5),
    generator.Mask(fraction=0.5))

G_true = generator.Generator(x0=x_GT, model=forward_model)
    
dataset_true = dataset.GeneratorWrapper(G_true)

# generator
G = generator.Generator(shape=x_GT.shape, model=forward_model)

# discriminator
D = discriminator.ConvMax(x_GT.size)

# training
G, D = train.train_WGAN(D, G, dataset_true,
                        num_steps=50, batch_size=3,
                        learning_rate_G=1e-3, learning_rate_D=1e-3,
                        num_steps_G=1, num_steps_D=1,
                        reg_weight=1e0, device=torch.device('cuda'),
                        out_folder=outdir)
