Modular arrangement of convolutional neural network on CIFAR10 dataset.
Master_Model.ipynb - Main python module. Different classes are called in below order from this module.
- S7_Datatransform.py -> This module stores data tranform class. There are 2 methods 

  a) test_transforms -> to create transform objects with just convert to tensor and normalize methods 
  
  b) train_transforms -> to create transform objects with before_normalization tranform otions and after_normalization methods 
  Returns a transform object based on the options and methods chosen
- S7_Dataloaders.py -> 
