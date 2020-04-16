import torch
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import dataset, generator, discriminator
# params

do_masking = True

if not do_masking:
    sigma = 0.5
    num_steps = 40
    num_steps_G = 3
    num_steps_D = 3

    batch_size = 3
    learning_rate_G = 1.0e-2
    learning_rate_D = 1.0e-2

    reg_weight = 1.0e1

    output_step = 5
else:
    sigma = 0.5
    num_steps = 500
    num_steps_G = 3
    num_steps_D = 3

    batch_size = 3
    learning_rate_G = 1.0e-3
    learning_rate_D = 1.0e-3

    reg_weight = 1.0e0
    output_step = 10
    
device = torch.device('cuda')

# setup output folder
outdir = 'results_' + time.strftime("%Y%m%d-%H%M%S")
os.makedirs(outdir)

# setup plots
plt.set_cmap('gray')

# true data distribution
im = scipy.misc.face()
im = np.mean(im/255, axis=2)
im = im[100:100+256, 400:400+256]
im = im - np.mean(im)

G_true = generator.NoisyImage(sigma=sigma, x0=im, do_masking=do_masking)
G_true.requires_grad_(False)
G_true = G_true.to(device)
data_true = iter(torch.utils.data.DataLoader(
    dataset.GeneratorWrapper(G_true), batch_size=batch_size))

# generator
G = generator.NoisyImage(size=im.shape, sigma=sigma, do_masking=do_masking)
G = G.to(device)
data_fake = iter(torch.utils.data.DataLoader(
    dataset.GeneratorWrapper(G), batch_size=batch_size))

# discriminator
D = discriminator.ConvMax(im.size)
    
D = D.to(device)

# training setup
loss_fcn = torch.nn.MSELoss(reduction='mean')
optim_G = torch.optim.Adam(G.parameters(), lr=learning_rate_G)
optim_D = torch.optim.Adam(D.parameters(), lr=learning_rate_D)

# training loop
print('%10s\t%10s\t%10s\t%10s\t%10s'
      % ('step', 'x-x_true', 'loss_G', 'loss_D', 'loss_D_reg'))
for step in range(num_steps):
    for inner_step in range(num_steps_G):
        G.requires_grad_(True)
        D.requires_grad_(False)
        
        x_fake = next(data_fake)

        loss_G = torch.mean(-D(x_fake))
        
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

    for inner_step in range(num_steps_D):
        G.requires_grad_(False)
        D.requires_grad_(True)

        x_true = next(data_true)
        x_fake = next(data_fake)

        score_true = torch.mean(D(x_true))
        score_fake = torch.mean(D(x_fake))

        loss_D = -score_true + score_fake

        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        # regularization
        with torch.no_grad():
             alpha = torch.rand(batch_size, device=device)
             alpha = alpha.unsqueeze(1).unsqueeze(2)
             x_between = alpha*x_true + (1-alpha)*x_fake

        # using backward leaks memory (I think),
        # so we use grad instead
        # see https://github.com/pytorch/pytorch/issues/4661
        x_between.requires_grad_(True)
        Dx = D(x_between)
        grad_x = torch.autograd.grad(
            Dx, x_between, torch.ones_like(Dx), create_graph=True)[0]
        reg_D = reg_weight * torch.mean(
            (
                torch.sqrt(torch.sum(grad_x**2, axis=(1,2)))
                -1)**2
        )
        optim_D.zero_grad()
        reg_D.backward()
        optim_D.step()     
             

    
    # print loss, make plots
    with torch.no_grad():
        print("%10d\t%10.3e\t%10.3e\t%10.3e\t%10.3e" %
              (step,
               loss_fcn(G.x, G_true.x).item(),
               loss_G.item(),
               loss_D.item(),
               reg_D.item()
               ))

        if step % output_step != 0:
            continue
        
        im_hat = G.x.detach().cpu().squeeze().numpy()
        vmin, vmax = im.min(), im.max()
        
        fig, ax = plt.subplots(1, 3)
        im_h = ax[0].imshow(im, vmin=vmin, vmax=vmax)
        ax[0].set_title('ground truth')
        fig.colorbar(im_h, ax=ax[0])
        
        im_h = ax[1].imshow(im_hat, vmin=vmin, vmax=vmax)
        ax[1].set_title('reconstruction')
        fig.colorbar(im_h, ax=ax[1])

        im_h = ax[2].imshow(np.abs(im-im_hat))
        ax[2].set_title('absolute error')
        fig.colorbar(im_h, ax=ax[2])

        fig.savefig(os.path.join(outdir,f'im_{step}.png'))    
        plt.close(fig)

        fig, ax = plt.subplots(2, batch_size, squeeze=False)
        for ind in range(batch_size):
            ax[0,ind].imshow(x_fake[ind].cpu())
            ax[1,ind].imshow(x_true[ind].cpu())

        ax[0,0].set_ylabel('fake', rotation=0)
        ax[1,0].set_ylabel('true', rotation=0)
        fig.tight_layout()
            
        fig.savefig(os.path.join(outdir, f'batch_{step}.png'))
        plt.close(fig)


# show results        
im_hat = G.x.detach().cpu().squeeze()
plt.imshow(im_hat)
plt.savefig('out.png')
