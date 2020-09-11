import torch
import os
import matplotlib.pyplot as plt
import logging
import pandas as pd

import dataset

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
    
    """
    plt.set_cmap('gray')

    # -----
    fig, ax = plt.subplots()

    ax.set_title('Ground Truth')
    im_h = ax.imshow(x_gt)
    fig.colorbar(im_h, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder,f'GroundTruth.png'))
    plt.close(fig)
    """
    
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

    #optim_G.param_groups[0]['lr'] *= .1 

    # main loop
    col_names = (
        'step', 'x-x_GT (MSE)', 'loss_G', 'D(true)',
        'D(fake)', 'loss_D_reg', 'Avg(MSE)')
    logging.info('%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s'
                 % col_names)
    history = pd.DataFrame(columns=col_names)

    count = 0
    rolling_sum = 0
    average = 1

    for step in range(num_steps):
        # update D
        for inner_step in range(num_steps_D):
            G.requires_grad_(False)
            D.requires_grad_(True)

            y_true = next(Y_true)
            y_fake = next(Y_fake)

            #print("\n THE SHAPE OF y_true IS:" + str(y_true.shape))
            rolling_sum += y_true.sum(dim=0) 
           # print(str(rolling_sum.shape) + " 999999999999")

            score_true = torch.mean(D(y_true))
            score_fake = torch.mean(D(y_fake))

            loss_D = -score_true + score_fake

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # regularization
            with torch.no_grad():
          #       alpha = torch.rand(batch_size, device=device)
               #  alpha = alpha.unsqueeze(1).unsqueeze(2)
                 #x_between = alpha*y_true + (1-alpha)*y_fake # just y_true 
                 x_between = y_true


            # using backward leaks memory (I think),
            # so we use grad instead
            # see https://github.com/pytorch/pytorch/issues/4661
            x_between.requires_grad_(True)

            Dx = D(x_between) 

            grad_x = torch.autograd.grad(Dx, x_between, torch.ones_like(Dx), create_graph=True)[0]  

            reg_D = (reg_weight / 2.0) * torch.mean(
            (torch.sqrt(torch.sum(grad_x**2, axis=(1,2))))**2
            )

            """   
            reg_D = reg_weight * torch.mean(
                (
                    torch.sqrt(torch.sum(grad_x**2, axis=(1,2)))
                    -1)**2 
            ) 
            """
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
            #   y_true = D.x.detach()
            x_hat = G.x.detach()
            mse = (x_hat - x_gt).pow(2).mean() 

            average = rolling_sum / ((step + 1) * batch_size * num_steps_D) 

            #average = rolling_sum #/ (step * num_steps_D)  
            if(step == 50):
                optim_G.param_groups[0]['lr'] *= .1

            count += 1
            
            AvgMSE = (average - x_gt).pow(2).mean()

            #  print("Avg: ", float(AvgMSE))
            current_history = (
                   step,
                   float(mse),
                   float(loss_G.item()),
                   float(score_true.item()),
                   float(score_fake.item()),
                   float(reg_D.item()),
                   float(AvgMSE)
                   )
            logging.info(
                "%10d\t%10.3e\t%10.3e\t%10.3e\t%10.3e\t%10.3e\t%10.3e" %
                current_history
            )

            history = history.append(pd.DataFrame([current_history], columns=col_names))

            if step % output_step != 0:
                continue
    return G, D, history

#scp -o ProxyCommand="ssh huwiomuh@scully.egr.msu.edu nc mhuwio@35.12.218.162:22"  mhuwio@35.12.218.162:~/GanTests/simple-WGAN/results /Users/moehuwio/MLtests/simple-WGAN/results

#scp  mhuwio@sai.dhcp.egr.msu.edu:~/GanTests/simple-WGAN/results/MULTI-SWEEP/FULL-DATA.npy ~/ResultStation/



#scp huwiomuh@arctic.cse.msu.edu:~/ProjectsCSE-320/Project1/proj01.tutorial ~/Desktop/ 
