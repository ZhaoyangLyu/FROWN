import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir, normalize=True):
        self.data_dir = data_dir
        data = torch.load(data_dir)
        self.X = data['X'].float() # of shape (num_samples, num_features)
        self.y = data['y']-1 # of shape num_samples
        self.mean = data['mean'].unsqueeze(0).float() # of shape (1, num_features)
        self.std = data['std'].unsqueeze(0).float() # of shape (1, num_features)

        if normalize:
            self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx]
