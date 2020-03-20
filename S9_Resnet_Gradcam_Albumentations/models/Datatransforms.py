from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np

# # class for Transformations ## 
class Transforms_custom:
      def __init__(self,normalize=False, mean=None, stdev=None):
            
          if normalize and (not mean or not stdev):
            raise ValueError('mean and stdev both are required for normalize transform')
            
          self.normalize = normalize
          self.mean      = mean      ## Make sure to pass the mean and stdev whenever normalization is set to true 
          self.stdev     = stdev
      
      
      # Define a method for test data set as it does not need extra transformations.
      # We just need to normalize with mean and stdev
      
      def test_transforms(self):
          transforms_list = []
          if (self.normalize):
             transforms_list.append(A.Normalize(self.mean,self.stdev))
            
          transforms_list.append = [AP.ToTensor()]
          self.transforms = A.Compose(transforms_list)
          
      # Define a method for train data . It can have multiple transformations other than changing to tensor and normalizing
      # so create your lists accordingly . One before normalization and one after it
      
      def train_transforms(self , before_norm=None, after_norm=None):
          transforms_list = []  
          if before_norm:
             transforms_list = before_norm 
             
          if (self.normalize):
             transforms_list.append(A.Normalize(self.mean,self.stdev))
           
          if after_norm:
             transforms_list.extend(after_norm)
          
          transforms_list.append(AP.ToTensor())
           
          self.transforms = A.Compose(transforms_list)
      
      def __call__(self, img):
          img = np.array(img)
          return self.transforms(image=img)['image']
