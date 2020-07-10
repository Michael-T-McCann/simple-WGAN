import torch
import os
import matplotlib.pyplot as plt
import numpy as np

import dataset
import sys
# save the output into the sam dir 
# check the logging package  
def train_WGAN(D, G, dataset_true,
               num_steps=10, batch_size=1,
               learning_rate_G=1.0, learning_rate_D=1.0,
               num_steps_G=1, num_steps_D=1,
               reg_weight=1.0,
               device=torch.device('cpu'), out_folder='',
               output_step=1, x_gt = []):

    # todo: this shouldn't go here, nor should all the plotting at the end
    # could I refactor with a yield?
    plt.set_cmap('gray')


    # ----- 

    fig, ax = plt.subplots()

    ax.set_title('Ground Truth')
    im_h = ax.imshow(x_gt)
    fig.colorbar(im_h, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder,f'GroundTruth.png'))
    plt.close(fig)
 
    x_gt = torch.tensor(x_gt,device=device)  


    D = D.to(device)
    G = G.to(device)

    dataset_true.to(device) 
 
    # setup
    Y_true = iter(torch.utils.data.DataLoader( 
        dataset_true, batch_size=batch_size))

    Y_fake = iter(torch.utils.data.DataLoader(
        dataset.GeneratorWrapper(G), batch_size=batch_size))

    loss_fcn = torch.nn.MSELoss(reduction='mean')
    optim_G = torch.optim.Adam(G.parameters(), lr=learning_rate_G)
    optim_D = torch.optim.Adam(D.parameters(), lr=learning_rate_D)

    resPath =  out_folder + '/' + "res.txt"
    sys.stdout = open(resPath, "w")
    # main loop
    print('%10s\t%10s\t%10s\t%10s\t%10s\t%10s'
          % ('step', 'x-x_GT', 'loss_G', 'D(true)', 'D(fake)', 'loss_D_reg')) 

    count = 0 
    rolling_sum = 1
    average = 1

    for step in range(num_steps):
        # update D
        for inner_step in range(num_steps_D):
            G.requires_grad_(False)
            D.requires_grad_(True)   

            y_true = next(Y_true)
            y_fake = next(Y_fake) 

            score_true = torch.mean(D(y_true))
            score_fake = torch.mean(D(y_fake)) 

            loss_D = -score_true + score_fake 

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # regularization
            with torch.no_grad():
                 alpha = torch.rand(batch_size, device=device)
                 alpha = alpha.unsqueeze(1).unsqueeze(2)
                 x_between = alpha*y_true + (1-alpha)*y_fake

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

        for inner_step in range(num_steps_G):
            G.requires_grad_(True)
            D.requires_grad_(False)

            y_fake = next(Y_fake)

            loss_G = torch.mean(-D(y_fake))

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()


        # print loss, make plots
        with torch.no_grad():
            x_hat = G.x.detach() 
         #   y_true = D.x.detach() 

            
            mse = (x_hat - x_gt).pow(2).mean()  
            
            rolling_sum += y_true 
            average = rolling_sum / (count * batch_size)  
            count += 1  

            print("Avg: ", average)
            print("%10d\t%10.3e\t%10.3e\t%10.3e\t%10.3e\t%10.3e" %
                  (step,
                   float(mse),
                   loss_G.item(),
                   score_true.item(),
                   score_fake.item(),  
                   reg_D.item(), 
                   )) 


            if step % output_step != 0:
                continue

            x_hat = G.x.detach().cpu().squeeze().numpy()
            #vmin, vmax = im.min(), im.max()
            #vmin, vmax = 0, 1
            
            fig, ax = plt.subplots()
            ax.set_title('reconstruction')
            im_h = ax.imshow(x_hat)
            fig.colorbar(im_h, ax=ax)

            #fig, ax = plt.subplots(1, 3, figsize=(8,2))
            #im_h = ax[0].imshow(im, vmin=vmin, vmax=vmax)
            #ax[0].set_title('ground truth')
            #fig.colorbar(im_h, ax=ax[0])

            #im_h = ax[1].imshow(x_hat, vmin=vmin, vmax=vmax)
            #ax[1].set_title('reconstruction')
            #fig.colorbar(im_h, ax=ax[1])

            #im_h = ax[2].imshow(abs(im-x_hat))
            #ax[2].set_title('absolute error')
            #fig.colorbar(im_h, ax=ax[2])

            fig.tight_layout()
            fig.savefig(os.path.join(out_folder,f'im_{step}.png'))
            plt.close(fig)

            num_plots = min(batch_size, 5)
            fig, ax = plt.subplots(2, num_plots, squeeze=False)
            for ind in range(num_plots):
                ax[0,ind].imshow(y_fake[ind].detach().cpu())
                ax[1,ind].imshow(y_true[ind].detach().cpu())

            ax[0,0].set_ylabel('fake', rotation=0)
            ax[1,0].set_ylabel('true', rotation=0)
            fig.tight_layout()

            fig.savefig(os.path.join(out_folder, f'batch_{step}.png'))
            plt.close(fig)

    return G, D


# scp -r /home/mhuwio/GanTests/simple-WGAN/results huwiomuh@scully.egr.msu.edu:~/ResultStation
#scp -o ProxyCommand="ssh huwiomuh@scully.egr.msu.edu nc mhuwio@35.12.218.162:22"  mhuwio@35.12.218.162:~/GanTests/simple-WGAN/results /Users/moehuwio/MLtests/simple-WGAN/results