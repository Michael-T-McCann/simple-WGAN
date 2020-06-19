import torch
import torch.nn as nn
import numpy as np



class ConvMax(nn.Module):
    """ based on VGG https://arxiv.org/pdf/1409.1556.pdf
    """
    def __init__(self, numel, conv_channels=32, linear_channels=32):
        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(conv_channels, conv_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),
        ])
  
        num_pooling = 2
        
        self.FC_layers = nn.ModuleList([
            nn.Linear(
                conv_channels*2 // 4**num_pooling * numel,
                linear_channels),
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
    

class DeepConvMax(nn.Module):
    """ 
    """
    def __init__(self, numel, conv_channels=32, linear_channels=32, num_layers=2):
        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)])


        for layer in range(num_layers-1):
            self.conv_layers.extend([
                nn.Conv2d(conv_channels*2**layer, conv_channels*2**(layer+1), kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)])


        
        self.FC_layers = nn.ModuleList([
            nn.Linear(
                conv_channels*2**(num_layers-1) // 4**num_layers * numel,
                linear_channels),
            nn.ReLU(),
            nn.Linear(linear_channels, linear_channels),
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
    
