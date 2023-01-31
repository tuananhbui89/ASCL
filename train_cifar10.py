import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18
from resnet import ResNet18

from utils import *
from utils_cm import * 
from mysetting import * 
from models import get_projection, Wrapper

mu = torch.tensor(CIFAR10_MEAN).view(3,1,1).cuda()
std = torch.tensor(CIFAR10_STD).view(3,1,1).cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

# -------- Path setting ----------
import os 
if os.path.exists('/trainman-mount/trainman-storage-3f66db6b-0e06-4bbb-8676-ecbf01ceab69/bta/AML/ASCL_pt/'):
    os.chdir('/trainman-mount/trainman-storage-3f66db6b-0e06-4bbb-8676-ecbf01ceab69/bta/AML/ASCL_pt/')
else:
    os.chdir('./')
WP = os.path.dirname(os.path.realpath('__file__')) + '/'
print(WP)

save_dir = WP + basedir + '/' + modeldir + '/'
mkdir_p(basedir)
mkdir_p(save_dir)

mkdir_p(save_dir+'/codes/')
backup('./', save_dir+'/codes/')
model_dir = save_dir + 'model.pt'
model_best_dir = save_dir + 'model_best.pt'

logfile = save_dir + 'log.txt'
writer = SummaryWriter(save_dir+'log/')


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(os.path.join(save_dir, 'output.log')),
        logging.StreamHandler()
    ])

logger.info(args)

np.random.seed(20212022)
torch.manual_seed(20212022)
torch.cuda.manual_seed(20212022)

transforms = [Crop(32, 32), FlipLR()]


dataset = cifar10('../data/')
train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
    dataset['train']['labels']))
train_set_x = Transform(train_set, transforms)
train_batches = Batches(train_set_x, args.bs, shuffle=True, set_random_choices=True, num_workers=2)

test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
test_batches = Batches(test_set, args.bs, shuffle=False, num_workers=2)

epsilon = args.epsilon
pgd_alpha = args.step_size

if args.model == 'preactresnet18':
    net = PreActResNet18()
elif args.model == 'resnet18':
    net = ResNet18()
elif args.model == 'wideresnet':
    net = WideResNet(34, 10, widen_factor=10, dropRate=0.0)
else:
    raise ValueError("Unknown model")

ProjLayer = get_projection(args.model, args.feat_dim)

model = Wrapper(CoreModel=net, NormLayer=None, ProjLayer=ProjLayer)

model = nn.DataParallel(model).cuda()
model.train()
params = model.parameters()

opt = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()

epochs = args.epochs

def lr_schedule(t):
    if t  < 100:
        return 0.1
    elif t < 105:
        return 0.01
    else:
        return 0.001


best_test_robust_acc = 0
best_val_robust_acc = 0
start_epoch = 0

logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
for epoch in range(start_epoch, epochs):
    model.train()
    start_time = time.time()
    train_loss = 0
    train_acc = 0
    train_robust_loss = 0
    train_robust_acc = 0
    train_n = 0
    for i, batch in enumerate(train_batches):

        X, y = batch['input'], batch['target']

        lr = lr_schedule(epoch + (i + 1) / len(train_batches))
        opt.param_groups[0].update(lr=lr)

        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.num_steps, 1, 'l_inf')
        delta = delta.detach()

        robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_loss = criterion(robust_output, y)

        opt.zero_grad()
        robust_loss.backward()
        opt.step()

        output = model(normalize(X))
        loss = criterion(output, y)

        train_robust_loss += robust_loss.item() * y.size(0)
        train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

    train_time = time.time()

    model.eval()
    test_loss = 0
    test_acc = 0
    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0
    for i, batch in enumerate(test_batches):
        X, y = batch['input'], batch['target']

        # Random initialization
        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.num_steps, 1, 'l_inf')
        delta = delta.detach()

        robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_loss = criterion(robust_output, y)

        output = model(normalize(X))
        loss = criterion(output, y)

        test_robust_loss += robust_loss.item() * y.size(0)
        test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        test_n += y.size(0)

    test_time = time.time()

    logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
        epoch, train_time - start_time, test_time - train_time, lr,
        train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
        test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

    # save checkpoint
    if (epoch+1) % 1 == 0 or epoch+1 == epochs:
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pt'))
        torch.save(opt.state_dict(), os.path.join(save_dir, f'opt_{epoch}.pt'))

    # save best
    if test_robust_acc/test_n > best_test_robust_acc:
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_best.pt'))
        best_test_robust_acc = test_robust_acc/test_n

        
