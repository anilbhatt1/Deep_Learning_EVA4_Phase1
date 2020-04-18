from torchvision import datasets

class Dataset(object):

    def gettraindataset(self, train_transforms):
        return datasets.CIFAR10(root='data', train=True,
                                download=True, transform=train_transforms)

    def gettestdataset(self, test_transforms):
        return datasets.CIFAR10(root='data', train=False,
                                download=True, transform=test_transforms)

    def getclassesinCIFAR10dataset(self=None):
        # specify the image classes
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

        return classes;
