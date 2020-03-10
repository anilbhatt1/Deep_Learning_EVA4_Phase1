from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# # class for Calculating and storing testing losses and testing accuracies of model for each epoch ## 
class Test_loss:

       def test_loss_calc(self,model, device, test_loader):
           self.model        = model
           self.device       = device
           self.test_loader  = test_loader  
       
           model.eval()
           
           correct        = 0
           processed      = 0    
           test_loss      = 0
           test_accuracy  = 0 
           test_losses    = []
           test_acc       = []
           
           with torch.no_grad():               # For test data, we won't do backprop, hence no need to capture gradients
                for images,labels in enumerate(test_loader):
                    images,labels    = images.to(device),labels.to(device)
                    labels_pred      = model(images)
                    test_loss        += F.nll_loss(labels_pred, labels, reduction = 'sum').item()                        
                    labels_pred_max  = labels_pred.argmax(dim =1, keepdim = True)
                    correct          += labels_pred_max.eq(labels.view_as(labels_pred_max)).sum().item()
                
                processed   = len(test_loader)
                test_loss   /= processed  # Calculating overall test loss for the epoch
                test_losses.append(test_loss)    
                  
                test_accuracy =  correct/processed
                test_acc.append(test_accuracy)
                  
                print('\nTest set: Average loss: {:.4f}, Test Accuracy: {:.2f}\n'.format(test_loss, test_accuracy))

           return test_losses, test_acc
