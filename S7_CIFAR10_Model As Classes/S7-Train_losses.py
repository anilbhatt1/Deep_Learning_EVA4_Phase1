from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# # class for Calculating and storing training losses and training accuracies of model for each batch per epoch ## 
class Train_loss:
      def __init__(self,model, device, train_loader, optimizer, epoch, factor):
          self.model        = model
          self.device       = device
          self.train_loader = train_loader
          self.optimizer    = optimizer
          self.epoch        = epoch
          self.factor       = factor
        
      def train_loss_calc(self):
          model.train()
          pbar = tqdm(train_loader)
          
          correct             = 0
          processed           = 0
          train_losses        = []
          train_acc           = []
          
          for batch_idx, (images, labels) in enumerate(pbar):
              images, labels = images.to(device), labels.to(device)   # Moving images and correspondig labels to GPU
              optimizer.zero_grad()  # Zeroing out gradients at start of each batch so that backpropagation won't take accumulated value
              labels_pred = model(images)  # Calling CNN model to predict the images
              loss = F.nll_loss(labels_pred, labels)   # Calculating Negative Likelihood Loss by comparing prediction vs ground truth
              
              # Applying L1 regularization to the training loss calculated
              L1_criterion = nn.L1Loss(size_average = None, reduce = None, reduction = 'mean')
              reg_loss     = 0
              for param in model.parameters():
                zero_tensor = torch.rand_like(param) * 0 # Creating a zero tensor with same size as param
                reg_loss    += L1_criterion(param, zero_tensor)
              loss += factor * reg_loss 
              train_losses.append(loss)
            
              # Backpropagation
              loss.backward()
              optimizer.step()
              
              # Calculating accuracies
              labels_pred_max = labels_pred.argmax(dim = 1, keepdim = True) # Getting the index of max log probablity predicted by model
              correct         += labels_pred_max.eq(labels.view_as(labels_pred_max)).sum().item() # Getting count of correctly predicted
              processed       += len(images) # Getting count of processed images
              train_acc_batch = (correct/processed)*100
              pbar.set_description(desc=f'Epoch = {epoch} Train Loss = {loss.item()} Batch Id = {batch_idx} Train Accuracy = {train_acc_batch:0.2f}')
              train_acc.append(train_acc_batch)  
             
          return train_losses, train_acc         
                
                
                
              
              
              
