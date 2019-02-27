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

class LSTM_Attention(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.0, dropout_method='pytorch'):
        super(LSTM_Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 5 * hidden_size)
        self.h2h_j = nn.Linear(hidden_size, 5 * hidden_size)
        self.h2h_i = nn.Linear(hidden_size, 5 * hidden_size)
        self.reset_parameters()
#         assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h_iprev_j,h_ijprev,c_iprev_j,c_ijprev = hidden
#         h = h[i,j].view(h.size(1), -1)
#         print(c_ijprev.shape)
        c_ijprev = c_ijprev.unsqueeze(1)
        c_iprev_j = c_iprev_j.unsqueeze(1)
        h_ijprev = h_ijprev.unsqueeze(1)
        h_iprev_j = h_iprev_j.unsqueeze(1)
#         x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h_i(h_ijprev) + self.h2h_j(h_iprev_j)

        # activations
        gates = torch.sigmoid(preact[:, :4 * self.hidden_size])
        g_t = torch.tanh(preact[:, 4 * self.hidden_size:])
        i_t = gates[:, :self.hidden_size]
        fl_t = gates[:, self.hidden_size:2 * self.hidden_size]
        fr_t = gates[:, 2 * self.hidden_size:3 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

#         # cell computations
#         if do_dropout and self.dropout_method == 'semeniuta':
#             g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_ij =  c_iprev_j * fl_t + c_ijprev * fr_t + i_t * g_t

#         if do_dropout and self.dropout_method == 'moon':
#                 c_t.data.set_(th.mul(c_t, self.mask).data)
#                 c_t.data *= 1.0/(1.0 - self.dropout)

        h_ij = torch.tanh(o_t * c_ij)

#         # Reshape for compatibility
#         if do_dropout:
#             if self.dropout_method == 'pytorch':
#                 F.dropout(h_ij, p=self.dropout, training=self.training, inplace=True)
#             if self.dropout_method == 'gal':
#                     h_t.data.set_(th.mul(h_t, self.mask).data)
#                     h_t.data *= 1.0/(1.0 - self.dropout)

        h_ij = h_ij.squeeze(1)
        c_ij = c_ij.squeeze(1)
        return h_ij, c_ij          
    

class CoarseClf(nn.Module):
    def __init__(self,n_class):
        super(CoarseClf,self).__init__()      
        model = ptr_models.squeezenet1_1(pretrained=True)
        clf_new = model.classifier
        in_ftrs = clf_new[1].in_channels
        out_ftrs = clf_new[1].out_channels
        clf_new = list(clf_new.children())
        clf_new[1] = nn.Conv2d(in_ftrs, n_class, kernel_size=(1, 1),stride=(1,1))
        clf_new[3] = nn.AvgPool2d(13, stride=1)
        child_count = 0
        for child in model.features.children():
            if child_count < 12:
                for param in child.parameters():
                    param.requires_grad = False
            child_count+=1
            
        self.features = model.features  
        self.classifier = nn.Sequential(*clf_new)
        self.num_classes = n_class
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  

#snet = CoarseClf(len(dsets_coarse["train"].coarse_class))

class FineClf_FC_layer(nn.Module):
    def __init__(self,in_ftrs,num_classes):
        super(FineClf_FC_layer,self).__init__()
        self.classifier = nn.Linear(in_ftrs**2,num_classes)

    def forward(self,x):
        batch_size = x.size()[0]
        x = x.view(batch_size, 512**2)
        return self.classifier(x)
        
class FineClf(nn.Module):
    def __init__(self):
        super(FineClf,self).__init__()
        model1 = ptr_models.squeezenet1_1(pretrained=True)
        model2 = ptr_models.resnet18(pretrained=True)
        child_count = 0
        for child in model1.features.children():
            if child_count < 12:
                for param in child.parameters():
                    param.requires_grad = False
            child_count+=1        
        child_count = 0
        for child in model2.children():
            if child_count < 8:
                for param in child.parameters():
                    param.requires_grad = False
            child_count+=1        
        self.classifier = {}
        in_ftrs = model2.fc.in_features
        model1 = list(model1.features.children())
        model1.append(nn.AvgPool2d(13, stride=1))
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*list(model2.children())[:-1])
        
    def forward(self,x):
        batch_size = x.size()[0]
        x1 = self.model1(x)
        x2 = self.model2(x)
        x1 = x1.view(batch_size,512,1)
        x2 = x2.view(batch_size,512,1)
        x = torch.bmm(x1,torch.transpose(x2,1,2))
        assert x.size() == (batch_size,512,512)
        x = torch.sqrt(x + 1e-5)
        x = F.normalize(x)
        return x
                                    
