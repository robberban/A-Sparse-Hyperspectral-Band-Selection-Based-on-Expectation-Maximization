import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage import io



def envi_normalize(imgData):
    # img_max =np.max(imgData, axis=2 ,keepdims = True)
    img_max =np.max(imgData,keepdims = True)
    return imgData / (img_max+0.0001)#65535#


class MyDataset_whole(torch.utils.data.Dataset):
    def __init__(self, train_set, dim = 144, feature_extraction = False, dataType = 'train'):
        self.dim = dim
        self.feature_extraction = feature_extraction
        train_set = '_cut_x1_64'#'_cut_x1_7'#
        self.dataType = dataType
        if dataType == "train":
            LabelPath = '/home/glk/datasets/HIC_dataset/2013_DFTC/label_' + dataType + train_set + '/'#label/#_4
            ImgRootPath = '/home/glk/datasets/HIC_dataset/2013_DFTC/image_'+ dataType + train_set + '/'#
            data_file = "train" + train_set + ".txt"
            self.imgpath = ImgRootPath
            data_file = '/home/glk/datasets/HIC_dataset/2013_DFTC/' + data_file
            self.label_path = LabelPath

        else:
            LabelPath = '/home/glk/datasets/HIC_dataset/2013_DFTC/label_' + dataType + train_set + '/'#label/#_4
            ImgRootPath = '/home/glk/datasets/HIC_dataset/2013_DFTC/image_'+ dataType + train_set + '/'#
            data_file = "test" + train_set + ".txt"  # Water_shenzhen
            self.imgpath = ImgRootPath
            self.label_path = LabelPath
            data_file = '/home/glk/datasets/HIC_dataset/2013_DFTC/' + data_file


        with open(data_file,'r') as f:
            dataFile = f.readlines()
        self.img_list = dataFile

    def __getitem__(self, index):
        if self.dataType == 'train':
            doc_dir = self.imgpath + self.img_list[index].split('.')[0] + '.npy'
            Image = np.load(doc_dir).transpose(2,0,1)

            label = io.imread(self.label_path + self.img_list[index].split('\n')[0]).astype(int)
        else:
            doc_dir = self.imgpath + self.img_list[index].split('.')[0] + '.npy'
            label = io.imread(self.label_path + self.img_list[index].split('\n')[0]).astype(int)
            Image = np.load(doc_dir).transpose(2,0,1)

        Image = envi_normalize(Image)
        return Image, label

    def __len__(self):
        return len(self.img_list)
