# Import the torch library
import torch 

# Class to create a Data Loader 
# Data Loader is a place holder using which we will load the train and test data sets.
class DataLoader:
      
    def __init__(self, shuffle, batch_size, seed):
        self.shuffle = shuffle,
        self.batch_size = batch_size,
        self.seed    = seed
         
        cuda = torch.cuda.is_available()
            
        if cuda:
           torch.cuda.manual_seed(seed) # Seed is for reproducibility
      
        # # dataloader arguments # # which we load
        self.dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=shuffle, batch_size=int(batch_size/2))

    def load(self, data):
        return torch.utils.data.DataLoader(data, **self.dataloader_args)
