Transforms used -> RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)

**Model - ResNet modified** 
        Uses this new ResNet Architecture for Cifar10 (Batch size = 512):
        
        PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        
        Layer1 -
        
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        
        R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        
        Add(X, R1)
        
        Layer 2 -
        
        Conv 3x3 [256k]
        
        MaxPooling2D
        
        BN
        
        ReLU
        
        Layer 3 -
        
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        
        R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        
        Add(X, R2)
        
        MaxPooling with Kernel Size 4
        
        FC Layer 
        
        SoftMax
        
**Implementing One Cycle Policy for LR hyperparameter tuning**
        Total Epochs = 24
        
        Max at Epoch = 5
        
        LRMIN = 1/10 * LRMAX
        
        LRMAX = Found Via. LR Range Test
        
        NO Annihilation which means will stop in 1 cycle (single triangle with no extension)
