import torch

class NoisyImage(torch.nn.Module):
    def __init__(self, size=None, sigma=0.0, x0=None):
        super().__init__()

        self.sigma = sigma
        
        if x0 is None:
            self.x = torch.nn.Parameter(torch.zeros(size))
        else:
            self.x = torch.nn.Parameter(torch.tensor(x0))

    def forward(self):
        x = self.x.clone()
        x = x + self.sigma * torch.randn_like(x)
        return x
                
