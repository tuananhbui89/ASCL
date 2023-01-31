"""
    Adversarial Training 
    Output: 
    - Pretrained model
    Args: 
    - ds: 'mnist', 'cifar10', 'cifar100'
    - model: 'cnn', 'resnet18', 'preactresnet18', 'wideresnet' 
"""
from pgd import pgd_loss
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

from mysetting import * 
from models import get_model, get_optimizer, adjust_learning_rate
from models import get_projection
from models import Normalize, Wrapper, Wrapper_with_LC, LinearClassifier
from utils_cm import mkdir_p, writelog, backup
from mytrain import train, test, adv_test, get_acc
from dataset import load_data
from pgd import pgd_attack

import re 
#------------------------------------------------------
# Loading dataset 
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

train_params['lccomw'] = args.lccomw
train_params['lcsmtw'] = args.lcsmtw
train_params['gbcomw'] = args.gbcomw
train_params['gbsmtw'] = args.gbsmtw
train_params['confw'] = args.confw
train_params['neg_type'] = args.neg_type
train_params['dist'] = args.dist
train_params['tau'] = args.tau
train_params['hidden_norm'] = args.hidden_norm

eval_params = train_params.copy()
eval_params['num_steps'] = 10
eval_params['epsilon'] = epsilon
# ------------------------------------------------------
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

model_dir = save_dir + 'model_with_lc.pt' 
model_best_dir = save_dir + 'model_with_lc_best.pt'
encoder_dir = save_dir + 'model.pt'
lc_dir = save_dir + 'lc.pt'

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
classifier = LinearClassifier(args.model, num_classes=num_classes)

encoder = Wrapper(CoreModel=net, NormLayer=Normalize(mean=mu, std=std), ProjLayer=ProjLayer)
model_with_lc = Wrapper_with_LC(CoreModel=net, NormLayer=Normalize(mean=mu, std=std), LC=classifier)

# Loading pretrain encoder 
state_dict = torch.load(encoder_dir, map_location='cpu')

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    encoder = encoder.cuda()
    classifier = classifier.cuda()

    encoder.load_state_dict(state_dict)

opt, lr = get_optimizer(ds=args.ds, model=classifier, architecture=args.model) # 
#------------------------------------------------------
# Train model 
pre_acc = -1. 

for epoch in range(args.epochs): 
    opt = adjust_learning_rate(opt, epoch, lr=lr, ds=args.ds)

    encoder.eval()
    classifier.train()

    num_batches = len(train_loader.dataset) // 128

    for batch_idx, (data, target) in enumerate(train_loader):
        assert(encoder.training is False)
        assert(classifier.training is True)

        data, target = data.to(device), target.to(device)
        opt.zero_grad()

        adv_x, _ = pgd_attack(model_with_lc, data, target, device, train_params, logadv=False, status='eval')

        encoder.eval()
        classifier.train()

        nat_output = model_with_lc(data)
        adv_output = model_with_lc(adv_x)

        # with torch.no_grad(): 
        #     _, nat_z = encoder(data, return_z=True)
        #     _, adv_z = encoder(adv_x, return_z=True)
        # nat_output = classifier(nat_z)
        # adv_output = classifier(adv_z)

        loss = F.cross_entropy(nat_output, target, reduction='none')
        loss += F.cross_entropy(adv_output, target, reduction='none')
        loss = torch.mean(loss)
        loss.backward()
        opt.step()

        nat_acc = get_acc(nat_output, target)

        if batch_idx % log_interval == 0:    
            writestr = [
                ('Train_iter={}', epoch*num_batches + batch_idx),
                ('loss={:.4f}', loss.item()), 
                ('nat_acc={:.4f}', nat_acc.item()), 

            ]
            writestr = '  ,'.join([t.format(v) for (t, v) in writestr]) 
            print(writestr)
            writer.add_scalar('loss', loss.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('nat_acc', nat_acc.item(), epoch*num_batches + batch_idx)    

    nat_acc = test(model_with_lc, test_loader, device, return_count=False, num_classes=num_classes)

    if epoch % log_period == 0 and epoch > 0:
        adv_acc = adv_test(model_with_lc, test_loader, device, eval_params, return_count=False, num_classes=num_classes)
        writelog('epoch:{}, nat_acc:{}, adv_acc:{}'.format(epoch, nat_acc, adv_acc), logfile)

        if adv_acc >= pre_acc: 
            pre_acc = adv_acc 
            torch.save(model_with_lc.state_dict(), model_best_dir)
            torch.save(classifier.state_dict(), lc_dir)
    else: 
        writelog('epoch:{}, nat_acc:{}'.format(epoch, nat_acc), logfile)

    torch.save(model_with_lc.state_dict(), model_dir)
    torch.save(classifier.state_dict(), lc_dir)

    writer.flush()
writer.close()   


nat_acc  = test(model_with_lc, test_loader, device, return_count=False, num_classes=num_classes)
adv_acc= adv_test(model_with_lc, test_loader, device, eval_params, return_count=False, num_classes=num_classes)
writelog('epoch:{}, nat_acc:{}, adv_acc:{}'.format(epoch, nat_acc, adv_acc), logfile)
