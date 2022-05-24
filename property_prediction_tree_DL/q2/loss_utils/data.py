from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    
    def __init__(self, x, y, TRUEY):
        self.x = x
        self.y = y
        self.TRUEY = TRUEY
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx,:], (~torch.isnan(self.y[idx,:])), self.TRUEY[idx,:]