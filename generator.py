import numpy as np
import torch
import torchvision

class NoisyImage(torch.nn.Module):
    def __init__(self, size=None, sigma=0.0, x0=None, do_crop=False):
        super().__init__()

        self.crop_size = 256
        self.do_crop = do_crop

        self.sigma = sigma
        
        if x0 is None:
            self.x = torch.nn.Parameter(torch.zeros(size))
        else:
            self.x = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float))

    def forward(self):
        x = self.x
        x = x + self.sigma * torch.randn_like(x)
        if self.do_crop:
            m = np.random.randint(0, self.x.shape[0]-self.crop_size)
            n = np.random.randint(0, self.x.shape[1]-self.crop_size)
            x = x[m:m+self.crop_size, n:n+self.crop_size]
                
        return x
