import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import cv2
import os
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
import functools
import torch
import pydicom
import vtk
from vtk.util import numpy_support


def get_training_augmentation(y=224,x=224):
    train_transform = [
                       albu.RandomBrightnessContrast(p=0.5),
                       albu.HorizontalFlip(p=0.5),
                       albu.ElasticTransform(p=0.2),
                       albu.GridDistortion(p=0.2),
                       albu.VerticalFlip(p=0.5),
                       albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=90, p=0.3, border_mode = cv2.BORDER_REPLICATE),
#                          albu.Transpose(p=0.5),
                       albu.OneOf([
                           albu.CenterCrop(224,224),
                           albu.RandomCrop(224,224),    
                       ],p=1.0),
                       ]
    return albu.Compose(train_transform)


formatted_settings = {
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],}


def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0
    return x

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_validation_augmentation(y=224,x=224):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.Resize(256, 256),albu.CenterCrop(224,224)]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')

class CTDataset2D(Dataset):
    def __init__(self,df,jpeg_dir,transforms = albu.Compose([albu.HorizontalFlip()]),preprocessing=None,size=256,mode='val'):
        self.df_main = df.values
        if mode=='val':
            self.df = self.df_main
        else:
            self.update_train_df()
        self.jpeg_dir = jpeg_dir
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.size=size


    def __getitem__(self, idx):
        row = self.df[idx]
        img = cv2.imread(glob.glob(f"{self.jpeg_dir}/{row[0]}/{row[1]}/*{row[2]}.jpg")[0])
        label = row[3:].astype(int)
#         label = label if label[0]==1 else np.zeros_like(label)
        label = label[0:1]
        if self.transforms:
            img = self.transforms(image=img)['image']
        if self.preprocessing:
            img = self.preprocessing(image=img)['image']
        return img,torch.from_numpy(label.reshape(-1))

    def __len__(self):
        return len(self.df)
    
    def update_train_df(self):
        df0 = self.df_main[self.df_main[:,3]==0]
        df1 = self.df_main[self.df_main[:,3]==1]
        np.random.shuffle(df0)
        self.df = np.concatenate([df0[0:len(df1)*2],df1],axis=0)
class CTDataset3D(Dataset):
    def __init__(self,df,jpeg_dir,transforms = None,preprocessing=None,size=256,mode='val'):
        self.df_main = df
        self.jpeg_dir = jpeg_dir
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.size=size

    def __getitem__(self, idx):
        mini = self.df_main[idx].values
        all_paths = [f"{self.jpeg_dir}/{row[0]}/{row[1]}/{row[-1]}_{row[2]}.jpg" for row in mini]
        img = [self.transforms(image=cv2.imread(p))['image'] for p in all_paths]
        label = mini[:,3:-1].astype(int)
        if self.preprocessing:
            img = [self.preprocessing(image=im)['image'] for im in img]
        return np.array(img),mini[0,0]

    def __len__(self):
        return len(self.df_main)
