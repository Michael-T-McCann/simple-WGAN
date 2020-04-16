import torch
import scipy.misc
import numpy as np

import dataset, generator
# params
sigma = 0.1
num_steps = 1000
#num_G_steps

batch_size = 10
learning_rate = 1.0e-2

# true data distribution
im = scipy.misc.face()
im = np.mean(im/255, axis=2)
im = im[100:100+256, 400:400+256]

G_true = generator.NoisyImage(sigma=sigma, x0=im)
G_true.requires_grad_(False)
data_true = iter(torch.utils.data.DataLoader(
    dataset.GeneratorWrapper(G_true), batch_size=batch_size))

# generator
G = generator.NoisyImage(size=im.shape, sigma=sigma)
data_fake = iter(torch.utils.data.DataLoader(
    dataset.GeneratorWrapper(G), batch_size=batch_size))


#D = discriminator.Basic()  # the discriminator

# training setup
loss_fcn = torch.nn.MSELoss(reduction='mean')
optim = torch.optim.Adam(G.parameters(), lr=learning_rate)

# training loop
for step in range(num_steps):
    x_true = next(data_true)
    x_fake = next(data_fake)

    loss = loss_fcn(x_true, x_fake)
    optim.zero_grad()
    loss.backward()
    optim.step()

    with torch.no_grad():
        print(loss_fcn(G.x, G_true.x))

    

        




