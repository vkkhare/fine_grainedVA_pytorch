import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import ImageFolder
import pandas as pd
import torchvision.models as ptr_models
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
def load_checkpoint(file,model,optimizer,best_prec1=None):
    if os.path.isfile(file):
        print("=> loading checkpoint '{}'".format(file))
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
#         best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(file, checkpoint['epoch']))
        return start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(file))
        return 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Dataset_Creator_IND():
    def __init__(self,root_dir,transform=None,test_tr_val='train',coarse=None):
        self.coarse_class = coarse
        self.root_dir = root_dir
        self.transform = transform
        self.test_tr_val = test_tr_val
        if self.coarse_class:
            self.dataset = ImageFolder(os.path.join(root_dir,self.test_tr_val,self.coarse_class), self.transform)
        else:    
            self.dataset = Coarse_Dataset(self.root_dir,self.transform,self.test_tr_val)
    def get_dataset(self):
        return self.dataset

class Coarse_Dataset(data.Dataset):
    
    def __init__(self,root_dir,transform=None, test_tr_val='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.test_tr_val = test_tr_val
        self.transform = transform
        self.coarse_class = os.listdir(os.path.join(root_dir,test_tr_val))
        self.class_maps = {i:k for k,i in enumerate(os.listdir(os.path.join(self.root_dir,self.test_tr_val)))}
        self.description_dict = {i:os.listdir(os.path.join(root_dir,test_tr_val,i)) for i in self.coarse_class}
        self.df = pd.read_csv(self.root_dir+'/'+test_tr_val+'_coarse.csv',sep='\t',index_col=0) 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name,coarse_label,fine_label = self.df.iloc[idx]
        image = cv2.imread(img_name)
#         print(img_name,image)
        if len(image.shape) != 3:
            image = cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)
        sample = {'feature': image, 'coarse_label': self.class_maps[coarse_label] ,'fine_label': fine_label}   
        if self.transform:
            sample['feature'] = self.transform(sample['feature'])
        return sample

