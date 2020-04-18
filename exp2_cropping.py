import time, os, shutil
import torch
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import dataset, generator, discriminator, train

# setup output directory
outdir = os.path.join(
    'results',
    'cropping_' + time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(outdir)

shutil.copyfile(__file__, os.path.join(outdir, 'exp.py'))


# true data distribution
x_GT = scipy.misc.face()
sz = tuple(m//2 for m in x_GT.shape[:2])[::-1]
x_GT = np.array(Image.fromarray(x_GT).resize(sz))
x_GT = np.mean(x_GT/255, axis=2)
x_GT = x_GT[50:50+150, 200:200+150]
x_GT = x_GT - np.mean(x_GT)

plt.set_cmap('gray')
fig, ax = plt.subplots()
ax.set_title('ground truth')
im_h = ax.imshow(x_GT)
fig.colorbar(im_h, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(outdir,f'im_gt.png'))    
plt.close(fig)


forward_model = torch.nn.Sequential(
    generator.RandomCrop(crop_shape=(128,128)),
    generator.AddNoise(sigma=0.1))

G_true = generator.Generator(x0=x_GT, model=forward_model)
    
dataset_true = dataset.GeneratorWrapper(G_true)

# generator
G = generator.Generator(shape=x_GT.shape, model=forward_model)

# discriminator
D = discriminator.ConvMax(
    next(dataset_true).numel(),
    conv_channels=64,
    linear_channels=128)

# training
G, D = train.train_WGAN(D, G, dataset_true,
                        num_steps=500, batch_size=2,
                        learning_rate_G=1e-3, learning_rate_D=1e-3,
                        num_steps_G=1, num_steps_D=1,
                        reg_weight=1e1, device=torch.device('cuda'),
                        out_folder=outdir, output_step=10)
