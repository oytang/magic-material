from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx,:]