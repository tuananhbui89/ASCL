import numpy as np
from numpy.testing._private.utils import requires_memory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from models import switch_status

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

"""
We follow the finding in Bag of Trick paper (Pang et al. 2020) to made some change as follow: 
- We init PGD attack with uniform noise (-epsilon, epsilon)
- In the original implementation, the model has been changed to evaluation stage (to change BN to evaluation). 
In this version, we do not change the model stage in the pgd attack. 
It means that: 
    - When training, the pgd attack will be generated with model in training stage 
    - When evaluation, the pgd attack will be generated with model in evaluation stage  
"""

# def pgd_loss_old(model,
#                 x_natural,
#                 y,
#                 device,
#                 optimizer,
#                 step_size=0.003,
#                 epsilon=0.031,
#                 perturb_steps=10,
#                 alpha=1.0,
#                 beta=1.0,
#                 projecting=True,
#                 distance='l_inf', 
#                 x_min=0.0, 
#                 x_max=1.0):
#     assert(beta == 1.0)
#     assert(distance == 'l_inf')
#     assert(projecting is True)
#     assert(x_max > x_min)

#     # model.eval()
#     model.train()
#     assert(model.training is True)

#     # random initialization 
#     x_adv = Variable(x_natural.data, requires_grad=True)
#     random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
#     x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

#     for _ in range(perturb_steps):
#         x_adv.requires_grad_()
#         with torch.enable_grad():
#             loss_ce = nn.CrossEntropyLoss(size_average=False)(model(x_adv), y) # Will not take average over batch 

#         grad = torch.autograd.grad(loss_ce, [x_adv])[0] # []
#         x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#         if projecting:
#             x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#         x_adv = torch.clamp(x_adv, x_min, x_max)

#     model.train()

#     x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
#     # zero gradient
#     optimizer.zero_grad()
#     # calculate robust loss
#     nat_output = model(x_natural)
#     adv_output = model(x_adv)

#     loss_natural = F.cross_entropy(nat_output, y, reduction='mean') # [b,]
#     loss_robust = F.cross_entropy(adv_output, y, reduction='mean') # [b, ]
        
#     loss = alpha * loss_natural + beta * loss_robust

#     log = dict()
#     log['loss_nat_ce'] = loss_natural
#     log['loss_adv_ce'] = loss_robust

#     return loss, x_adv, log 

def pgd_attack_old(model, X, y, device, attack_params, logadv=False, status='train'): 
    """
        Reference: 
            https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py 
            L2 attack: https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
        Args: 
            model: pretrained model 
            X: input tensor
            y: input target 
            attack_params:
                loss_type: 'ce', 'kl' or 'mart'
                epsilon: attack boundary
                step_size: attack step size 
                num_steps: number attack step 
                order: norm order (norm l2 or linf)
                random_init: random starting point 
                x_min, x_max: range of data 
    """
    # model.eval()

    # assert(attack_params['random_init'] == True)
    # assert(attack_params['projecting'] == True)
    assert(attack_params['order'] == np.inf)

    X_adv = Variable(X.data, requires_grad=True)

    if attack_params['random_init']:
        random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-attack_params['epsilon'], 
                                                            attack_params['epsilon']).to(device)
        X_adv = Variable(X_adv.data + random_noise, requires_grad=True)
    
    X_adves = []
    for _ in range(attack_params['num_steps']):
        with torch.enable_grad():
            if attack_params['loss_type'] == 'ce':
                loss = F.cross_entropy(model(X_adv), y, reduction='sum')
            elif attack_params['loss_type'] == 'kl': 
                loss = F.kl_div(F.log_softmax(model(X_adv), dim=1), 
                                    F.softmax(model(X), dim=1), 
                                    reduction='sum')

        loss.backward()
        eta = attack_params['step_size'] * X_adv.grad.data.sign()
        X_adv = Variable(X_adv.data + eta, requires_grad=True)
        eta = torch.clamp(X_adv.data - X.data, 
                            -attack_params['epsilon'], 
                            attack_params['epsilon'])
        X_adv = Variable(X.data + eta, requires_grad=True)
        X_adv = Variable(torch.clamp(X_adv, 
                            attack_params['x_min'], 
                            attack_params['x_max']), requires_grad=True)

        if logadv:
            X_adves.append(X_adv)

    # switch_status(model, status)

    X_adv = Variable(X_adv.data, requires_grad=False)
    return X_adv, X_adves


def pgd_attack(model, X, y, device, attack_params, logadv=False, status='train'):
    
    restarts = 1
    norm = 'l_inf'
    early_stop = False 
    mixup = False
    X_adves = []

    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-attack_params['epsilon'], attack_params['epsilon'])
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*attack_params['epsilon']
        else:
            raise ValueError
        delta = clamp(delta, attack_params['x_min']-X, attack_params['x_max']-X)
        delta.requires_grad = True
        for _ in range(attack_params['num_steps']):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if attack_params['loss_type'] == 'ce':
                loss = F.cross_entropy(output, y, reduction='sum')
            elif attack_params['loss_type'] == 'kl':
                loss = F.kl_div(F.log_softmax(output, dim=1), 
                                    F.softmax(model(X), dim=1), 
                                    reduction='sum')
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + attack_params['step_size'] * torch.sign(g), 
                                min=-attack_params['epsilon'], 
                                max=attack_params['epsilon'])
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*attack_params['step_size']).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=attack_params['epsilon']).view_as(d)
            d = clamp(d, attack_params['x_min'] - x, attack_params['x_max'] - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        if attack_params['loss_type'] == 'ce':
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        elif attack_params['loss_type'] == 'kl':
            all_loss = F.kl_div(F.log_softmax(model(X+delta), dim=1), 
                                    F.softmax(model(X), dim=1), 
                                    reduction='sum')
        
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

        X_adv = X+max_delta.detach()

        if logadv:
            X_adves.append(X_adv)
    
    return X_adv, X_adves


def pgd_loss(model,
                x_natural,
                y,
                device,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                alpha=1.0,
                beta=1.0,
                projecting=True,
                distance='l_inf', 
                x_min=0.0, 
                x_max=1.0):
    assert(beta == 1.0)
    assert(distance == 'l_inf')
    assert(projecting is True)
    assert(x_max > x_min)

    # model.eval()
    model.train()
    assert(model.training is True)
    
    attack_params = {
        'epsilon': epsilon,
        'step_size': step_size,
        'num_steps': perturb_steps,
        'random_init': True, 
        'order': np.inf, 
        'loss_type': 'ce',
        'x_min': x_min,
        'x_max': x_max,
    }

    x_adv, _ = pgd_attack(model, x_natural, y, device, attack_params, logadv=False, status='train')

    model.train()

    x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    nat_output = model(x_natural)
    adv_output = model(x_adv)

    loss_natural = F.cross_entropy(nat_output, y, reduction='mean') # [b,]
    loss_robust = F.cross_entropy(adv_output, y, reduction='mean') # [b, ]
        
    loss = alpha * loss_natural + beta * loss_robust

    log = dict()
    log['loss_nat_ce'] = loss_natural
    log['loss_adv_ce'] = loss_robust

    return loss, x_adv, log 