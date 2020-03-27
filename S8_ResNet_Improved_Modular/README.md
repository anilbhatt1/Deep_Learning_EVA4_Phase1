Convolutional neural network (ResNet-18 Architecture) on CIFAR10 dataset.

**S8_Master Model_ResNet_CIFAR10.ipynb** - Main python module. Different classes are called in below order from this module. Model architecture used is ResNet-18. Model is not overfitting as we included various data augmentation transforms. ResNet-18 is a deeper architecture and hence to avoid overfitting more variety of images are required while training. More transforms helped in providing this variety thereby avaoiding overfitting.

- Datatransform.py -> This module stores data tranform class. There are 2 methods 

  a) test_transforms -> to create transform objects with just convert to tensor and normalize methods 
  
  b) train_transforms -> to create transform objects with before_normalization tranform otions and after_normalization methods 
  Returns a transform object based on the options and methods chosen
  
  test_transforms are used while downloading test data and train_transforms while downloading train data.
- Dataloaders.py -> This module creates data load class with shuffle=True, use cuda device if available etc. In this module 'load' method is called seperately to create testloader and trainloader instances by supplying test data and train data downloaded using transforms. Returns trainloader/testloader which will be used while training & testing the model epoch-by-epoch.
- Model.py -> This module hold the ResNet-18 model. Called inside Train losses with model.train() option and inside test losses with model.eval() option.
- Train_Losses.py -> This module is for calculating training loss & training accuracy & fine-tuning model using back-propagation on a batch-by-batch basis & epoch by epoch. Returns train loss and training accuracy for no: of epochs trained.
- Test_losses.py -> This module is for calculating testing loss & testing accuracy on an epoch-by-epoch basis.Returns test loss and test accuracy for no: of epochs trained.
