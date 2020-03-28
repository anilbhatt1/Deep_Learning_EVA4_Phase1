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
 
           label_dict     = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
           label_total    = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
           label_correct  = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}   
                       
           with torch.no_grad():               # For test data, we won't do backprop, hence no need to capture gradients
                for images,labels in test_loader:    # We are working in GPU, so 1 iteration will process 128 images(batch_size) in a go. Total 10,000/128 = 79 iterations will happen
 
                    images,labels    = images.to(device),labels.to(device)
                    labels_pred      = model(images)                                                            # Tensor with shape torch.Size([128, 10]) 
                    test_loss        += F.nll_loss(labels_pred, labels, reduction = 'sum').item()               # Use torch.Tensor.item() to get a Python number from a tensor containing a single value               
                    labels_pred_max  = labels_pred.argmax(dim =1, keepdim = True)                               # Tensor with shape torch.Size([128, 1]). We are taking maximum value out of 10 from 'pred' tensor
                    correct          += labels_pred_max.eq(labels.view_as(labels_pred_max)).sum().item()        # labels -> Tensor with shape torch.Size([128]). We are changing shape of labels to ([128, 1]) for comparison purpose
                    total            += labels.size(0)                                                          # Taking number of images in each batch size and accumulating it to get total images at end. Here labels.size(0)  = 128
                    
                    ''' labels_pred_max will look like below: torch.Size([128, 1])
                     ([[3],
                       [0],
                       [5],
                       .
                       .
                       [7]], device='cuda:0') -> 128th element
                       
                       labels will look like below: torch.Size([128])
                       ([3, 2, 5, 5, 0, 9,.....4, 4], device='cuda:0') 
                       
                      * labels_pred_max.item() -> This will fail because torch.Tensor.item() is to get a Python number from a tensor containing a single value   
                      * labels.item() ->  This will fail because torch.Tensor.item() is to get a Python number from a tensor containing a single value
                      * labels_pred.item() ->  This will fail because torch.Tensor.item() is to get a Python number from a tensor containing a single value
                      * labels.view_as(labels_pred_max).item() -> This will fail because torch.Tensor.item() is to get a Python number from a tensor containing a single value
                      * if labels_pred_max == labels:  -> This will fail beacuse we are comparing different shapes                   
                      * if labels_pred_max[2] == labels[2]: -> This will work because we are gathering specific elements and comparing
                      * len(labels_pred_max) = 128 which is same as batch_size
                    '''
                                 
                    for i in range(len(labels_pred_max)):
                        counter_key = ' '
                        counter_key = label_dict.get(labels[i].item())  
                        label_total[counter_key] += 1 
                            
                        if labels_pred_max[i] == labels[i]:
                           label_correct[counter_key] += 1  
                        else:    
                           if count_wrong   < 26 and current_epoch == (total_epoch - 1):     # Capturing 26 wrongly predicted images for last epoch
                              wrong_predict.append(images[i])                                # with its predicted and actual class 
                              predicted_class.append(labels_pred_max[i].item())
                              actual_class.append(labels[i].item())
                              count_wrong += 1
                                                 
              
                test_loss   /= total  # Calculating overall test loss for the epoch
                test_losses.append(test_loss)    
                                  
                test_accuracy =  (correct/total)* 100
                test_acc.append(test_accuracy)             
               
                print('\nTest set: Average loss: {:.4f}, Test Accuracy: {:.2f}\n' .format(test_loss, test_accuracy))

           return test_losses, test_acc, wrong_predict, predicted_class, actual_class, label_total, label_correct
