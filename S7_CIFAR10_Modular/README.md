Modular arrangement of convolutional neural network on CIFAR10 dataset.
Master_Model.ipynb - Main python module. Different classes are called in below order from this module.
- S7_Datatransform.py -> This module stores data tranform class. There are 2 methods 

  a) test_transforms -> to create transform objects with just convert to tensor and normalize methods 
  
  b) train_transforms -> to create transform objects with before_normalization tranform otions and after_normalization methods 
  Returns a transform object based on the options and methods chosen
  
  test_transforms are used while downloading test data and train_transforms while downloading train data.
- S7_Dataloaders.py -> This module creates data load class with shuffle=True, use cuda device if available etc. In this module 'load' method is called seperately to create testloader and trainloader instances by supplying test data and train data downloaded using transforms.
- 
