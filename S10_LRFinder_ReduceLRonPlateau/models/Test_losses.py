from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# # class for Calculating and storing testing losses and testing accuracies of model for each epoch ## 
class Test_loss:

       def test_loss_calc(self,model, device, test_loader, total_epoch, current_epoch):
           self.model        = model
           self.device       = device
           self.test_loader  = test_loader
           self.total_epoch  = total_epoch
           self.current_epoch= current_epoch   
       
           model.eval()
           
           correct        = 0 
           total          = 0              
           test_loss      = 0
           test_accuracy  = 0 
           test_losses    = []
           test_acc       = []
           predicted_class= []
           actual_class   = []
           wrong_predict  = []
           count_wrong    = 0 
           classwise_acc  = []
           
           with torch.no_grad():               # For test data, we won't do backprop, hence no need to capture gradients
                for images,labels in test_loader:
 
                    images,labels    = images.to(device),labels.to(device)
                    labels_pred      = model(images)
                    test_loss        += F.nll_loss(labels_pred, labels, reduction = 'sum').item()                        
                    labels_pred_max  = labels_pred.argmax(dim =1, keepdim = True)
                    correct          += labels_pred_max.eq(labels.view_as(labels_pred_max)).sum().item()
                    total            += labels.size(0) 
              
                    #if labels_pred_max == labels: 
                    #   if labels.item() == 1:
                    #        print
                    
                    for i in range(len(labels_pred_max)):                        
                        if labels_pred_max[i] != labels[i]:
                           if count_wrong   < 5 and current_epoch == (total_epoch - 1):     # Capturing 26 wrongly predicted images for last epoch
                              wrong_predict.append(images[i])                                # with its predicted and actual class 
                              predicted_class.append(labels_pred_max[i].item())
                              actual_class.append(labels[i].item())
                              count_wrong += 1
                              print('count_wrong:',count_wrong)
                              print('labels_pred_max[i].shape :', labels_pred_max[i])
                              print('labels_pred_max[i] :', labels_pred_max[i])
                              print('labels[i].shape :', labels[i].shape)
                              print('labels[i]:', labels[i])     
                              x1 = labels[i].item()
                              print('x1:',x1)
                              print('type(x1):',type(x1))     
                
                test_loss   /= total  # Calculating overall test loss for the epoch
                test_losses.append(test_loss)    
                                  
                test_accuracy =  (correct/total)* 100
                test_acc.append(test_accuracy)             
               
                print('\nTest set: Average loss: {:.4f}, Test Accuracy: {:.2f}\n' .format(test_loss, test_accuracy))

           return test_losses, test_acc, wrong_predict, predicted_class, actual_class
