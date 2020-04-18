import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
# from albumentations.pytorch import ToTensor
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets
from torchvision.transforms import ToTensor
import requests
from tqdm import tqdm
from zipfile import ZipFile
import os.path
from os import path

class Utils:

    # helper function to un-normalize and display an image
    @staticmethod
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))
        # plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    def imshowt(tensor):
        tensor = tensor.squeeze()
        if len(tensor.shape) > 2:
            tensor = tensor.permute(1, 2, 0)
        img = tensor.cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.show()

    def printdatetime(self):
        print("Model execution started at:" + datetime.datetime.today().ctime())

    # def printgpuinfo():
    #     gpu_info = "nvidia-smi"
    #     # gpu_info = '\n'.join(gpu_info)
    #     print(gpu_info)

    def calculate_mean_std_deviation_cifar10(self=None):

        traindataset = datasets.CIFAR10('./data', train=True, download=True, transform=ToTensor())
        testdataset = datasets.CIFAR10('./data', train=False, download=True, transform=ToTensor())

        data = np.concatenate([traindataset.data, testdataset.data], axis=0)
        data = data.astype(np.float32) / 255.

        means = []
        stdevs = []

        for i in range(3):  # 3 channels
            pixels = data[:, :, :, i].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        return [means[0], means[1], means[2]], [stdevs[0], stdevs[1], stdevs[2]]

    def calculate_mean_std_deviation(train_data, test_data):

        data = np.concatenate([train_data, test_data], axis=0)
        data = data.astype(np.float32) / 255.

        means = []
        stdevs = []

        for i in range(3):  # 3 channels
            pixels = data[:, :, :, i].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        return [means[0], means[1], means[2]], [stdevs[0], stdevs[1], stdevs[2]]

    def download_file(folder_path, url):

        # get the file name
        file_name = url.split("/")[-1]
        folder_path = folder_path + "/" + file_name

        if path.exists(folder_path):
            print('File: {} already downloaded.'.format(file_name))
            return folder_path

        # read 1024 bytes every time
        buffer_size = 1024
        # download the body of response by chunk, not immediately
        response = requests.get(url, stream=True)

        # get the total file size
        file_size = int(response.headers.get("Content-Length", 0))

        # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
        progress = tqdm(response.iter_content(buffer_size), f"Downloading {folder_path}", total=file_size, unit="B",
                        unit_scale=True, unit_divisor=1024)
        with open(folder_path, "wb") as f:
            for data in progress:
                # write data read to the file
                f.write(data)
                # update the progress bar manually
                progress.update(len(data))

        return folder_path

    def extract_zip_file(file_path, extract_path):

        file_name = file_path.split("/")[-1]
        file_name = file_name.split(".")[0]
        output_folder = extract_path + '/' + file_name

        if path.exists(output_folder):
            print('File: {} already extracted.'.format(file_name))
            return output_folder

        print('Extracting file from {} to {}'.format(file_path, extract_path))
        with ZipFile(file_path, 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall(extract_path)

        print('File extraction completed.')
        return output_folder
