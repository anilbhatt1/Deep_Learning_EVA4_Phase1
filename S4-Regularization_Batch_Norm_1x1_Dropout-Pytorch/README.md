Model Architecture details
--------------------------
Block 1
-------

Used 3x3 convolutions with more number of channels (32) in initial block so that more feature maps can be created. Used padding = 1 to retain channel size.

-Used 1x1 convolution to combine the 32 channels to 16 channels. 1x1 helped to maintain parameters < 20K
Applied batch normalization to 16 channels. As it is MNIST dataset, may not have much effect. But will be useful for more complex datasets.
Applied 2x2, stride 2 maxpooling to reduce the channel size.
Applied dropout of 0.08 for regularization.
Block 2
-------
Used 3x3 convolutions on 16 channels and increased it to 32 layers in second layer so that more feature maps can be created. Used padding = 1.
Used 1x1 convolution to combine the 32 channels to 16 channels. 
Applied batch normalization to 16 channels.
Applied 2x2, stride 2 maxpooling to reduce the channel size.
Applied dropout of 0.08 for regularization.
Block 3
-------
Used 3x3 convolutions on 16 channels without padding
Used 3x3 convolutions on 16 channels without padding
Block 4
-------
Applied GAP using AdaptiveAvgPool2d to get 16 values back from 16 channels
Converted these 16 values to a 1D array using View method
Connected these 16 values to a fully connected layer of 10 nodes. 10 nodes was choosen because of 10 classes that we want to predict (digits :0-9).
Fully connected layer output converted to 1D array of 10 elements.
These 10 elements are passed to log_softmax for prediction.Used log_softmax as it is computationally efficient compared to softmax.

**Note: Bias was not used in any of the CNN layers.

Results
-------
No: of model parameters : 19,968
Trained the model for 19 epochs.
Maximum test accuracy of 99.46% achieved on 11th epoch
