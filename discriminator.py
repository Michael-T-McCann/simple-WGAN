import torch
import torch.nn as nn
import numpy as np

class ConvMax(nn.Module):
    """ based on VGG https://arxiv.org/pdf/1409.1556.pdf
    """
    def __init__(self, numel):
        super().__init__()

        num_channels = 32
        linear_channels = 32

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),
            ])

        self.FC_layers = nn.ModuleList([
            nn.Linear(num_channels*2 // 16 * numel, linear_channels),
            nn.ReLU(),
            nn.Linear(linear_channels, 1)
            ])

    def forward(self, x):
        x = x.unsqueeze(1)  # add channels
        for layer in self.conv_layers:
            x = layer(x)

        x = x.reshape(x.shape[0], -1)  # vectorize, but leave batch dim
        for layer in self.FC_layers:
            x = layer(x)
        
        return x
    
