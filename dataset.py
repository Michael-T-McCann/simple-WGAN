import torch

class GeneratorWrapper(torch.utils.data.IterableDataset):
    def __init__(self, G):
        self.G = G

    def __next__(self):
        return self.G()

    def __iter__(self):
        return self

    def to(self, device):
        self.G = self.G.to(device)
