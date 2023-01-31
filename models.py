import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from collections import OrderedDict

from small_cnn import * 
from resnet import * 
from preactresnet import *
from wideresnet import * 

def swish(x):
    return x * F.sigmoid(x)

def activ(key): 
    if key == 'relu': 
        return nn.ReLU() 
    elif key == 'elu': 
        return nn.ELU()
    elif key == 'swish': 
        return swish

def get_model(ds, model, activation=nn.ReLU()): 
    assert(ds == 'cifar10')
    if model == 'cnn': 
        return Cifar10(activation=activation)
    elif model == 'resnet18': 
        return ResNet18()
    elif model == 'preactresnet18': 
        return PreActResNet18()
    elif model == 'wideresnet': 
        return WideResNet()

"""
We tried with different training setting and report the PGD-AT performances (ResNet18, CIFAR10) as below: 
- (Ref. Zhang et al. 2020) (lr=0.1, m=0.9, wd=5e-4) 120 epochs, change at {60, 90, 110}. 
- (Ref. Zhang et al. 2019a) (lr=0.01, m=0.9, wd=5e-4) 105 epochs, change at {75, 90, 100}. 
- (Ref. Rice et al. 2020) (lr=0.1, m=0.9, wd=5e-4) 200 epochs, change at {100, 150}. 
- (Ref. Bag of tricks, default, Pang et al. 2020) (lr=0.1, m=0.9, wd=5e-4) 110 epochs, change at {100, 105}. 

Finally, we choose the setting as in Pang et al. 2020. More specifically: 
- optimizer: SGD with momentum 0.9, weight decay 5e-4
- learning rate scheduler: init with 0.1, learning rate decay with rate 0.1 at epoch {100, 105}
- training length: 110 epochs. 
"""

def adjust_learning_rate(optimizer, epoch, lr, ds): 
    assert(ds == 'cifar10')
    _lr = lr
    if epoch >= 100:
        _lr = lr * 0.1
    if epoch >= 105:
        _lr = lr * 0.01
    if epoch >= 110:
        _lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr
    return optimizer


def get_optimizer(ds, model, architecture):
    assert (ds == 'cifar10')

    lr = 0.1
    momentum = 0.9 
    weight_decay = 5e-4 
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    return opt, lr

def switch_status(model, status): 
    if status == 'train': 
        model.train()
    elif status == 'eval': 
        model.eval()
    else: 
        raise ValueError
#------------------------------------------------------
class DataWithIndex(Dataset):
    def __init__(self, train_data):
        self.data = train_data
        
    def __getitem__(self, index):
        data, target = self.data[index]
        
        return data, target, index

    def __len__(self):
        return len(self.data)

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).cuda()
        self.std = torch.tensor(std, dtype=torch.float32).cuda()

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

# class ProjHead(nn.Module): 
#     def __init__(self, dim_in, feat_dim=128):
#         super(ProjHead, self).__init__()
#         self.head = nn.Sequential(
#                 nn.Linear(dim_in, dim_in),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(dim_in, feat_dim)
#             )
    
#     def forward(self, x): 
#         # return F.normalize(self.head(x), dim=1)
#         return self.head(x) 

class ProjHead(nn.Module): 
    def __init__(self, dim_in, feat_dim=128):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(dim_in, feat_dim)
            )
    
    def forward(self, x): 
        # return F.normalize(self.head(x), dim=1)
        return self.head(x) 

model_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'wideresnet': 640, # NEED CHECK
}

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, model, num_classes=10):
        super(LinearClassifier, self).__init__()
        assert(model in model_dict.keys())
        feat_dim = model_dict[model]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

def get_projection(model, feat_dim=128): 
    assert(model in model_dict.keys())
    dim_in = model_dict[model]
    if feat_dim > 0:
        return ProjHead(dim_in, feat_dim=feat_dim)
    else: 
        return None

class Wrapper(nn.Module): 
    def __init__(self, CoreModel, NormLayer, ProjLayer=None):
        super(Wrapper, self).__init__()
        self.CoreModel = CoreModel 
        self.NormLayer = NormLayer
        self.ProjLayer = ProjLayer
    
    def forward(self, x, return_z=False): 
        if self.NormLayer is not None:
            xn = self.NormLayer(x)
        else: 
            xn = x 
        if return_z:
            output, z = self.CoreModel(xn, return_z=return_z)
            if self.ProjLayer is not None: 
                zp = self.ProjLayer(z)
                return output, zp
            else: 
                return output, z
        else:
            return self.CoreModel(xn, return_z=return_z)

class Wrapper_with_LC(nn.Module): 
    def __init__(self, CoreModel, NormLayer, LC):
        super(Wrapper_with_LC, self).__init__()
        self.CoreModel = CoreModel
        self.NormLayer = NormLayer
        self.LC = LC 
    
    def forward(self, x, return_z=False): 
        # assert(return_z is True) # Always use LC, therefore, always requires return_z 
        xn = self.NormLayer(x)
        _, z = self.CoreModel(xn, return_z=True)
        output = self.LC(z)

        if return_z: 
            return output, z 
        else: 
            return output


