import numpy as np
import torch
import torchvision

class NoisyImage(torch.nn.Module):
    def __init__(self, size=None, sigma=0.0, x0=None, do_masking=False):
        super().__init__()

        self.do_masking = do_masking

        self.sigma = sigma
        
        if x0 is None:
            self.x = torch.nn.Parameter(torch.zeros(size))
        else:
            self.x = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float))
                
    def forward(self):
        self.zero = torch.zeros_like(self.x)  # todo: make once, put on GPU
        x = self.x
        if self.do_masking:
            mask = torch.rand_like(x) > 0.5
            x = torch.where(mask, x, self.zero)

        x = x + self.sigma * torch.randn_like(x)
                
        return x
