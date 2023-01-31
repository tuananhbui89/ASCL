"""
    Adversarial Training 
    Output: 
    - Pretrained model
    Args: 
    - ds: 'mnist', 'cifar10', 'cifar100'
    - model: 'cnn', 'resnet18', 'preactresnet18', 'wideresnet' 
"""
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter

from mysetting import * 
from models import get_model, get_optimizer, adjust_learning_rate
from models import get_projection
from models import Normalize, Wrapper
from utils_cm import mkdir_p, writelog, backup
from mytrain import train, test, adv_test
from mytrain import adv_train
from dataset import load_data, load_two_transforms

import re 
#------------------------------------------------------
# Loading dataset 
if args.defense in ['scl_nat_2trans', 'ascl_pgd_2trans', 'ascl_trades_2trans']:
    train_loader, test_loader = load_two_transforms(ds=args.ds, 
                                train_batch_size=args.bs, 
                                test_batch_size=args.bs)

else:
    train_loader, test_loader = load_data(ds=args.ds, 
                                train_batch_size=args.bs, 
                                test_batch_size=args.bs)


# Parameter setting for Adversarial Training 
if args.ds == 'mnist':
    num_classes = 10
    x_max = 1.
    x_min = 0.
    epsilon = 0.3 
    step_size = 0.01
    num_steps= 10
    log_period = 1
    mu = MNIST_MEAN
    std = MNIST_STD

elif args.ds == 'cifar10': 
    num_classes = 10
    x_max = 1.
    x_min = 0.
    epsilon = 0.031
    step_size = 0.007
    num_steps= 10
    log_period = 1
    mu = CIFAR10_MEAN
    std = CIFAR10_STD

elif args.ds == 'cifar100': 
    num_classes = 100
    x_max = 1.
    x_min = 0.
    epsilon = 0.01
    step_size = 0.001
    num_steps= 10
    log_period = 1
    mu = CIFAR100_MEAN
    std = CIFAF100_STD

#------------------------------------------------------
# Params setting 
log_interval = 10

train_params = dict()

train_params['epsilon'] = args.epsilon # For training with custom epsilon 
train_params['step_size'] = step_size 
train_params['num_steps'] = num_steps
train_params['x_min'] = x_min
train_params['x_max'] = x_max

train_params['defense'] = args.defense
train_params['order'] = int(args.order) if args.order != 'inf' else np.inf
train_params['loss_type'] = args.loss_type
train_params['random_init'] = args.random_init
train_params['projecting'] = args.projecting
train_params['trades_beta'] = args.trades_beta
train_params['alpha'] = args.alpha

train_params['lccomw'] = args.lccomw
train_params['lcsmtw'] = args.lcsmtw
train_params['gbcomw'] = args.gbcomw
train_params['gbsmtw'] = args.gbsmtw
train_params['confw'] = args.confw
train_params['neg_type'] = args.neg_type
train_params['dist'] = args.dist
train_params['tau'] = args.tau
train_params['hidden_norm'] = args.hidden_norm
train_params['combine_type'] = args.combine_type

eval_params = train_params.copy()
eval_params['num_steps'] = 10
eval_params['epsilon'] = epsilon
# ------------------------------------------------------
import os 
if os.path.exists('/trainman-mount/trainman-storage-3f66db6b-0e06-4bbb-8676-ecbf01ceab69/bta/AML/ASCL_pt/'):
    # os.chdir('/trainman-mount/trainman-storage-3f66db6b-0e06-4bbb-8676-ecbf01ceab69/bta/AML/ASCL_pt/')
    WP = '/trainman-mount/trainman-storage-3f66db6b-0e06-4bbb-8676-ecbf01ceab69/bta/AML/ASCL_pt/'
else:
    # os.chdir('./')
    WP = './'
# WP = os.path.dirname(os.path.realpath('__file__')) + '/'
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

for key in train_params.keys(): 
    writelog('train_params, {}:{}'.format(key, train_params[key]), logfile)

torch.manual_seed(20212022)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
   

#------------------------------------------------------
# Model 
net = get_model(args.ds, args.model)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  net = nn.DataParallel(net)

ProjLayer = get_projection(args.model, args.feat_dim)

model = Wrapper(CoreModel=net, NormLayer=Normalize(mean=mu, std=std), ProjLayer=ProjLayer)
model.to(device)

opt, lr = get_optimizer(ds=args.ds, model=model, architecture=args.model)
#------------------------------------------------------
# Train model 
pre_acc = -1. 

for epoch in range(args.epochs): 
    opt = adjust_learning_rate(opt, epoch, lr=lr, ds=args.ds)

    if args.defense == 'none':
        writer = train(model, train_loader, epoch, opt, device, log_interval, train_params, writer)
    else:
        writer = adv_train(model, train_loader, epoch, opt, device, log_interval, train_params, writer)

    nat_acc = test(model, test_loader, device, return_count=False, num_classes=num_classes)
    if epoch % log_period == 0 and epoch > 0:
        adv_acc = adv_test(model, test_loader, device, eval_params, return_count=False, num_classes=num_classes)
        writelog('epoch:{}, nat_acc:{}, adv_acc:{}'.format(epoch, nat_acc, adv_acc), logfile)

        if adv_acc >= pre_acc: 
            pre_acc = adv_acc 
            torch.save(model.state_dict(), model_best_dir)
    else: 
        writelog('epoch:{}, nat_acc:{}'.format(epoch, nat_acc), logfile)

    torch.save(model.state_dict(), model_dir)
    if epoch % args.save_freq == 0:
        torch.save(model.state_dict(),
                    os.path.join(save_dir, 'model-nn-epoch{}.pt'.format(epoch)))
    writer.flush()
writer.close()   


nat_acc  = test(model, test_loader, device, return_count=False, num_classes=num_classes)
adv_acc= adv_test(model, test_loader, device, eval_params, return_count=False, num_classes=num_classes)
writelog('epoch:{}, nat_acc:{}, adv_acc:{}'.format(epoch, nat_acc, adv_acc), logfile)
