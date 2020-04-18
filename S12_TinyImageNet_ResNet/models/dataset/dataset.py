from torchvision import datasets

class Dataset(object):

    def gettraindataset(self, train_transforms):
        return datasets.CIFAR10(root='data', train=True,
                                download=True, transform=train_transforms)

    def gettestdataset(self, test_transforms):
        return datasets.CIFAR10(root='data', train=False,
                                download=True, transform=test_transforms)

    def get_tiny_imagenet_train_dataset(self, train_transforms, train_image_data, train_image_labels):
        from models.dataset import TinyImagenetDataset
        return TinyImagenetDataset(image_data=train_image_data, image_labels=train_image_labels,
                                   transform=train_transforms)

    def get_tiny_imagenet_test_dataset(self, test_transforms, test_image_data, test_image_labels):
        from models.dataset import TinyImagenetDataset
        return TinyImagenetDataset(image_data=test_image_data, image_labels=test_image_labels,
                                   transform=test_transforms)

    def getclassesinCIFAR10dataset(self=None):
        # specify the image classes
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

        return classes;
