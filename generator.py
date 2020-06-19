import numpy as np
import torch
import torchvision

"""
Q: Why are these modules when they could be functions?
A: It makes it easy to chain them together and also allows more parameters 
to be learned than just x.

"""

class Generator(torch.nn.Module):
    def __init__(self, shape=None, x0=None, model=None):
        super().__init__()

        self.model = model
        
        if x0 is None:
            self.x = torch.nn.Parameter(torch.zeros(shape))
        else:
            self.x = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float))
    
    def forward(self):
        return self.model(self.x)

class AddNoise(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
                
    def forward(self, x):
        return x + self.sigma * torch.randn_like(x)

    
class Mask(torch.nn.Module):
    def __init__(self, fraction=0.5):
        super().__init__()
        self.fraction = fraction

    def forward(self, x):
        mask = torch.rand_like(x) > self.fraction
        return x * mask

class RandomCrop(torch.nn.Module):
    def __init__(self, crop_shape):
        super().__init__()
        self.crop_shape = crop_shape

    def forward(self, x):
        m = np.random.randint(0, x.shape[0]-self.crop_shape[0]+1)
        n = np.random.randint(0, x.shape[1]-self.crop_shape[1]+1)
        return x[m:m+self.crop_shape[0], n:n+self.crop_shape[1]]        
        
    
