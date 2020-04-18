import matplotlib.pyplot as plt
import numpy as np
import torch

import models.utils.Utils as utils

class PlotData:

    def showImagesfromdataset(dataiterator, classes=None, values=None, image_count=20, col=2):
        images, labels = dataiterator.next()
        images = images.numpy()  # convert images to numpy for display

        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(25, 4))
        # display images
        for idx in np.arange(image_count):
            ax = fig.add_subplot(2, image_count / 2, idx + 1, xticks=[], yticks=[])
            if classes is not None:
                ax.set_title(classes[labels[idx]].strip())
            else:
                class_name = ''
                if "," in values[labels[idx]]:
                    class_name = values[labels[idx]].split(',')[0]
                else:
                    class_name = values[labels[idx]]
                ax.set_title(class_name.strip())
            utils.Utils.imshow(images[idx])

        #plt.savefig("images/imagesfromdataset.png", bbox_inches='tight')
