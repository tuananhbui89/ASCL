import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from contrastive_losses_v2 import soft_lcscl
from scl_loss import SupConLoss 
from compact_loss_pt import kl_loss_with_logits, my_norm
from pgd import pgd_attack
from utils import add_loss 

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def ascl_pgd(model,
                x_natural,
                y,
                device,
                optimizer,
                dist='linf',
                neg_type='leaking',
                temperature=0.07,
                soft_label = False,
                hidden_norm = False,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=10,
                alpha=1.0,
                beta=1.0,
                projecting=True,
                distance='l_inf', 
                x_min=0.0, 
                x_max=1.0, 
                lccomw=1.0, lcsmtw=1.0,
                gbcomw=1.0, gbsmtw=1.0, 
                confw=1.0, 
                combine_type=1):

    assert(alpha in [0.0, 1.0])
    assert(beta == 1.0)
    assert(distance == 'l_inf')
    assert(projecting is True)
    assert(x_max > x_min)
    assert(lccomw == 0.0)
    # assert(lcsmtw == 0.0)
    # assert(gbcomw > 0.0)
    assert(gbsmtw == 0.0)
    assert(confw == 0.0)

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
    assert(model.training is True)
    assert(x_adv.requires_grad is False)
    optimizer.zero_grad()

    # calculate robust loss
    nat_output, nat_z = model(x_natural, return_z=True)
    adv_output, adv_z = model(x_adv, return_z=True)

    # Cross entropy loss 
    loss_natural = F.cross_entropy(nat_output, y, reduction='mean') 
    loss_robust = F.cross_entropy(adv_output, y, reduction='mean') 
    
    num_classes = nat_output.shape[1]
    _l1 = F.one_hot(torch.argmax(nat_output, dim=-1), num_classes=num_classes) 
    _l2 = F.one_hot(torch.argmax(adv_output, dim=-1), num_classes=num_classes)
    preds = torch.cat([_l1, _l2], dim=0)

    if soft_label: 
        labels_concat = preds
    else: 
        y_onehot = F.one_hot(y, num_classes=num_classes)
        labels_concat = torch.cat([y_onehot, y_onehot], dim=0)

    # Supervised Contrastive Loss 
    loss_ascl = soft_lcscl(hidden=torch.cat([nat_z, adv_z], dim=0).double(), 
                        labels=labels_concat.double(), 
                        preds=preds.double(), 
                        dist=dist, 
                        neg_type=neg_type, 
                        hidden_norm=hidden_norm, 
                        temperature=temperature)
    
    # nat_z = torch.unsqueeze(nat_z, dim=1)
    # adv_z = torch.unsqueeze(adv_z, dim=1)
    # features = torch.cat([nat_z, adv_z], dim=1)
    # loss_ascl = SupConLoss()(features, labels=y)

    loss_vat = kl_loss_with_logits(adv_output, nat_output, reduction='mean')
    loss_com = torch.mean(my_norm(adv_z, nat_z, p=dist, normalize=hidden_norm))

    loss = gbcomw*loss_ascl 
    loss += add_loss(loss_vat, lcsmtw)
    loss += add_loss(loss_com, lccomw)
    loss += add_loss(loss_natural, alpha)
    loss += add_loss(loss_robust, beta)

    log = dict()
    log['loss_nat_ce'] = loss_natural
    log['loss_adv_ce'] = loss_robust
    log['loss_ascl'] = loss_ascl
    log['loss_vat'] = loss_vat
    log['loss_com'] = loss_com

    return loss, x_adv, log


def scl_pgd(model,
                x_natural,
                y,
                device,
                optimizer,
                dist='linf',
                neg_type='leaking',
                temperature=0.07,
                soft_label = False,
                hidden_norm = False,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=10,
                alpha=1.0,
                beta=1.0,
                projecting=True,
                distance='l_inf', 
                x_min=0.0, 
                x_max=1.0, 
                lccomw=1.0, lcsmtw=1.0,
                gbcomw=1.0, gbsmtw=1.0, 
                confw=1.0, 
                combine_type=1):

    assert(alpha in [0.0, 1.0])
    assert(beta == 1.0)
    assert(distance == 'l_inf')
    assert(projecting is True)
    assert(x_max > x_min)
    assert(lccomw == 0.0)
    # assert(lcsmtw == 0.0)
    # assert(gbcomw > 0.0)
    assert(gbsmtw == 0.0)
    assert(confw == 0.0)

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
    assert(model.training is True)
    assert(x_adv.requires_grad is False)
    optimizer.zero_grad()
    # calculate robust loss
    nat_output, nat_z = model(x_natural, return_z=True)
    adv_output, adv_z = model(x_adv, return_z=True)

    # Cross entropy loss 
    loss_natural = F.cross_entropy(nat_output, y, reduction='mean') # [b,]
    loss_robust = F.cross_entropy(adv_output, y, reduction='mean') # [b, ]
    
    nat_z = torch.unsqueeze(nat_z, dim=1)
    adv_z = torch.unsqueeze(adv_z, dim=1)
    features = torch.cat([nat_z, adv_z], dim=1)
    loss_ascl = SupConLoss()(features, labels=y)

    loss_vat = kl_loss_with_logits(adv_output, nat_output, reduction='mean')
    loss_com = torch.mean(my_norm(adv_z, nat_z, p=dist, normalize=hidden_norm))

    loss = gbcomw*loss_ascl 
    loss += add_loss(loss_vat, lcsmtw)
    loss += add_loss(loss_com, lccomw)
    loss += add_loss(loss_natural, alpha)
    loss += add_loss(loss_robust, beta)

    log = dict()
    log['loss_nat_ce'] = loss_natural
    log['loss_adv_ce'] = loss_robust
    log['loss_ascl'] = loss_ascl
    log['loss_vat'] = loss_vat
    log['loss_com'] = loss_com

    return loss, x_adv, log

def ascl_trades(model,
                x_natural,
                y,
                device,
                optimizer,
                dist='linf',
                neg_type='leaking',
                soft_label = False,
                hidden_norm=False,
                temperature=0.07,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=10,
                alpha=1.0,
                beta=1.0,
                projecting=True,
                distance='l_inf', 
                x_min=0.0, 
                x_max=1.0, 
                lccomw=1.0, lcsmtw=1.0,
                gbcomw=1.0, gbsmtw=1.0, 
                confw=1.0, 
                combine_type=1):
    # define KL-loss
    assert(alpha in [0.0, 1.0])
    assert(projecting is True)
    assert(distance == 'l_inf')
    assert(x_max > x_min)
    assert(lccomw == 0.0)
    # assert(lcsmtw == 0.0)
    # assert(gbcomw > 0.0)
    assert(gbsmtw == 0.0)
    assert(confw == 0.0)

    criterion_kl = nn.KLDivLoss(size_average=False)
    
    attack_params = {
        'epsilon': epsilon,
        'step_size': step_size,
        'num_steps': perturb_steps,
        'random_init': True, 
        'order': np.inf, 
        'loss_type': 'kl',
        'x_min': x_min,
        'x_max': x_max,
    }
    x_adv, _ = pgd_attack(model, x_natural, y, device, attack_params, logadv=False, status='train')
    assert(model.training is True)
    assert(x_adv.requires_grad is False)
    optimizer.zero_grad()

    # calculate robust loss
    nat_output, nat_z = model(x_natural, return_z=True)
    adv_output, adv_z = model(x_adv, return_z=True)
    batch_size = nat_output.shape[0]
    loss_natural = F.cross_entropy(nat_output, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_output, dim=1),
                                                    F.softmax(nat_output, dim=1))

    num_classes = nat_output.shape[1]
    _l1 = F.one_hot(torch.argmax(nat_output, dim=-1), num_classes=num_classes) 
    _l2 = F.one_hot(torch.argmax(adv_output, dim=-1), num_classes=num_classes)
    preds = torch.cat([_l1, _l2], dim=0)

    if soft_label: 
        labels_concat = preds
    else: 
        y_onehot = F.one_hot(y, num_classes=num_classes)
        labels_concat = torch.cat([y_onehot, y_onehot], dim=0)

    # Supervised Contrastive Loss 
    loss_ascl = soft_lcscl(hidden=torch.cat([nat_z, adv_z], dim=0).double(), 
                        labels=labels_concat.double(), 
                        preds=preds.double(), 
                        dist=dist, 
                        neg_type=neg_type, 
                        hidden_norm=hidden_norm, 
                        temperature=temperature)

    loss_vat = kl_loss_with_logits(adv_output, nat_output, reduction='mean')
    loss_com = torch.mean(my_norm(adv_z, nat_z, p=dist, normalize=hidden_norm))

    loss = gbcomw*loss_ascl 
    loss += add_loss(loss_vat, lcsmtw)
    loss += add_loss(loss_com, lccomw)
    loss += add_loss(loss_natural, alpha)
    loss += add_loss(loss_robust, beta)

    log = dict()
    log['loss_nat_ce'] = loss_natural
    log['loss_adv_ce'] = loss_robust
    log['loss_ascl'] = loss_ascl
    log['loss_vat'] = loss_vat
    log['loss_com'] = loss_com

    return loss, x_adv, log


def scl_nat_2trans(model,
                x_natural,
                y,
                device, 
                optimizer,
                **kwargs):

    """
        Natural Supervised Contrastive Learning 
        (Original implementation, do not use any adversarial examples)
        Just reuse the original implementation 
        Args: 
        - x_natural: inputs [2b, C, H, W]
        - y: labels [2b, ]
    """
    # model.train()
    assert(model.training is True)

    _, features = model(x_natural, return_z=True)
    batch_size = features.shape[0] // 2
    _, labels = torch.split(y, batch_size, dim=0)
    features_1, features_2 =  torch.split(features, batch_size, dim=0)
    features_1 = torch.unsqueeze(features_1, dim=1)
    features_2 = torch.unsqueeze(features_2, dim=1)
    features = torch.cat([features_1, features_2], dim=1) # [b,2,d]
    loss_scl = SupConLoss()(features, labels=labels)
    loss = loss_scl 

    log = dict()
    log['loss_scl'] = loss_scl

    return loss, x_natural, log 


def ascl_pgd_2trans(model,
                x_natural,
                y,
                device,
                optimizer,
                dist='linf',
                neg_type='leaking',
                temperature=0.07,
                soft_label = False,
                hidden_norm = False,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=10,
                alpha=1.0,
                beta=1.0,
                projecting=True,
                distance='l_inf', 
                x_min=0.0, 
                x_max=1.0, 
                lccomw=1.0, lcsmtw=1.0,
                gbcomw=1.0, gbsmtw=1.0, 
                confw=1.0, 
                combine_type=1):
    """
    Adversarial Supervised Contrastive Learning with two transformations 
    Args: 
        - x_two: inputs [2b, C, H, W] or (x1, x2)
        - y_two: labels [2b, ] or (y, y)
    Using adversary like pgd-attack, we have two adversarial examples (x1a, x2a) w.r.t. 
    input pair (x1, x2). With the encoder, we then have four latent code (z1, z2, z1a, z2a) 
    There are several ways to contrust the losses with one loss function 
        # option 1: total = ascl(cat[z1,z2], cat[z1a, z2a])
        # option 2: total = 0.5 * ascl(z1, z2) + 0.5 * ascl(z1a, z2a)  
        # option 3: total = 0.5 * ascl(z1, z1a) + 0.5 * ascl(z2, z2a)
    
    We tried with 3 options (all with same "leaking" ASCL version) and report the results as below: 
        # 


    """
    assert(alpha in [0.0, 1.0])
    assert(beta == 1.0)
    assert(distance == 'l_inf')
    assert(projecting is True)
    assert(x_max > x_min)
    assert(lccomw == 0.0)
    # assert(lcsmtw == 0.0)
    # assert(gbcomw > 0.0)
    assert(gbsmtw == 0.0)
    assert(confw == 0.0)
    assert(combine_type in [1,2,3])

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
    assert(model.training is True)
    assert(x_adv.requires_grad is False)
    optimizer.zero_grad()

    # calculate robust loss
    nat_output, nat_z = model(x_natural, return_z=True)
    adv_output, adv_z = model(x_adv, return_z=True)

    # Cross entropy loss 
    loss_natural = F.cross_entropy(nat_output, y, reduction='mean')
    loss_robust = F.cross_entropy(adv_output, y, reduction='mean') 
    
    num_classes = nat_output.shape[1]
    batch_size = nat_output.shape[0] // 2 

    _l1 = F.one_hot(torch.argmax(nat_output, dim=-1), num_classes=num_classes) 
    _l2 = F.one_hot(torch.argmax(adv_output, dim=-1), num_classes=num_classes)
    preds = torch.cat([_l1, _l2], dim=0) # [4b, num_classes]

    if soft_label: 
        labels_concat = preds # [4b, num_classes]
    else: 
        y_onehot = F.one_hot(y, num_classes=num_classes)
        labels_concat = torch.cat([y_onehot, y_onehot], dim=0) # [4b, num_classes]

    
    # option 1: total = ascl(cat[z1,z2], cat[z1a, z2a])
    if combine_type == 1:
        loss_ascl = soft_lcscl(hidden=torch.cat([nat_z, adv_z], dim=0).double(), 
                        labels=labels_concat.double(), 
                        preds=preds.double(), 
                        dist=dist, 
                        neg_type=neg_type, 
                        hidden_norm=hidden_norm, 
                        temperature=temperature)

    # ---------------------------------------------------------
    elif combine_type == 2: 
        # option 2: total = 0.5 * ascl(z1, z2) + 0.5 * ascl(z1a, z2a)
        p1, p2, p1a, p2a = torch.split(preds, batch_size, dim=0) # 4 pred labels, each shape: [b, num_classes]
        nat_z1, nat_z2 = torch.split(nat_z, batch_size, dim=0) # 2 nat-z, each shape [b, dim]
        adv_z1, adv_z2 = torch.split(adv_z, batch_size, dim=0) # 2 adv-z, each shape [b, dim]
        l1, l2, l1a, l2a = torch.split(labels_concat, batch_size, dim=0) # 4 true labels, each shape: [b, num_classes]

        loss_ascl = 0.5 * soft_lcscl(hidden=torch.cat([nat_z1, nat_z2], dim=0).double(), 
                            labels=torch.cat([l1, l2], dim=0).double(), 
                            preds=torch.cat([p1, p2], dim=0).double(), 
                            dist=dist, 
                            neg_type=neg_type, 
                            hidden_norm=hidden_norm, 
                            temperature=temperature)

        loss_ascl += 0.5 * soft_lcscl(hidden=torch.cat([adv_z1, adv_z2], dim=0).double(), 
                            labels=torch.cat([l1a, l2a], dim=0).double(), 
                            preds=torch.cat([p1a, p2a], dim=0).double(), 
                            dist=dist, 
                            neg_type=neg_type, 
                            hidden_norm=hidden_norm, 
                            temperature=temperature)
    # ---------------------------------------------------------

    elif combine_type == 3: 
        # option 3: total = 0.5 * ascl(z1, z1a) + 0.5 * ascl(z2, z2a)
        p1, p2, p1a, p2a = torch.split(preds, batch_size, dim=0) # 4 pred labels, each shape: [b, num_classes]
        nat_z1, nat_z2 = torch.split(nat_z, batch_size, dim=0) # 2 nat-z, each shape [b, dim]
        adv_z1, adv_z2 = torch.split(adv_z, batch_size, dim=0) # 2 adv-z, each shape [b, dim]
        l1, l2, l1a, l2a = torch.split(labels_concat, batch_size, dim=0) # 4 true labels, each shape: [b, num_classes]

        loss_ascl = 0.5 * soft_lcscl(hidden=torch.cat([nat_z1, adv_z1], dim=0).double(), 
                            labels=torch.cat([l1, l1a], dim=0).double(), 
                            preds=torch.cat([p1, p1a], dim=0).double(), 
                            dist=dist, 
                            neg_type=neg_type, 
                            hidden_norm=hidden_norm, 
                            temperature=temperature)

        loss_ascl += 0.5 * soft_lcscl(hidden=torch.cat([nat_z2, adv_z2], dim=0).double(), 
                            labels=torch.cat([l2, l2a], dim=0).double(), 
                            preds=torch.cat([p2, p2a], dim=0).double(), 
                            dist=dist, 
                            neg_type=neg_type, 
                            hidden_norm=hidden_norm, 
                            temperature=temperature)          

    loss_vat = kl_loss_with_logits(adv_output, nat_output, reduction='mean')
    loss_com = torch.mean(my_norm(adv_z, nat_z, p=dist, normalize=hidden_norm))

    loss = gbcomw*loss_ascl 
    loss += add_loss(loss_vat, lcsmtw)
    loss += add_loss(loss_com, lccomw)
    loss += add_loss(loss_natural, alpha)
    loss += add_loss(loss_robust, beta)

    log = dict()
    log['loss_nat_ce'] = loss_natural
    log['loss_adv_ce'] = loss_robust
    log['loss_ascl'] = loss_ascl
    log['loss_vat'] = loss_vat
    log['loss_com'] = loss_com

    return loss, x_adv, log



def ascl_trades_2trans(model,
                x_natural,
                y,
                device,
                optimizer,
                dist='linf',
                neg_type='leaking',
                temperature=0.07,
                soft_label = False,
                hidden_norm = False,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=10,
                alpha=1.0,
                beta=1.0,
                projecting=True,
                distance='l_inf', 
                x_min=0.0, 
                x_max=1.0, 
                lccomw=1.0, lcsmtw=1.0,
                gbcomw=1.0, gbsmtw=1.0, 
                confw=1.0, 
                combine_type=1):
    """
    Adversarial Supervised Contrastive Learning with two transformations 
    Args: 
        - x_two: inputs [2b, C, H, W] or (x1, x2)
        - y_two: labels [2b, ] or (y, y)
    Using adversary like pgd-attack, we have two adversarial examples (x1a, x2a) w.r.t. 
    input pair (x1, x2). With the encoder, we then have four latent code (z1, z2, z1a, z2a) 
    There are several ways to contrust the losses with one loss function 
        # option 1: total = ascl(cat[z1,z2], cat[z1a, z2a])
        # option 2: total = 0.5 * ascl(z1, z2) + 0.5 * ascl(z1a, z2a)  
        # option 3: total = 0.5 * ascl(z1, z1a) + 0.5 * ascl(z2, z2a)
    
    We tried with 3 options (all with same "leaking" ASCL version) and report the results as below: 
        # 


    """
    assert(alpha in [0.0, 1.0])
    assert(distance == 'l_inf')
    assert(projecting is True)
    assert(x_max > x_min)
    assert(lccomw == 0.0)
    # assert(lcsmtw == 0.0)
    # assert(gbcomw > 0.0)
    assert(gbsmtw == 0.0)
    assert(confw == 0.0)
    assert(combine_type in [1,2,3])

    attack_params = {
        'epsilon': epsilon,
        'step_size': step_size,
        'num_steps': perturb_steps,
        'random_init': True, 
        'order': np.inf, 
        'loss_type': 'kl',
        'x_min': x_min,
        'x_max': x_max,
    }
    x_adv, _ = pgd_attack(model, x_natural, y, device, attack_params, logadv=False, status='train')
    assert(model.training is True)
    assert(x_adv.requires_grad is False)
    optimizer.zero_grad()

    # calculate robust loss
    nat_output, nat_z = model(x_natural, return_z=True)
    adv_output, adv_z = model(x_adv, return_z=True)

    # Cross entropy loss 
    criterion_kl = nn.KLDivLoss(size_average=None, reduction='mean') 
    loss_natural = F.cross_entropy(nat_output, y, reduction='mean')
    loss_robust = criterion_kl(F.log_softmax(adv_output, dim=1),
                                F.softmax(nat_output, dim=1))
    
    num_classes = nat_output.shape[1]
    batch_size = nat_output.shape[0] // 2 

    _l1 = F.one_hot(torch.argmax(nat_output, dim=-1), num_classes=num_classes) 
    _l2 = F.one_hot(torch.argmax(adv_output, dim=-1), num_classes=num_classes)
    preds = torch.cat([_l1, _l2], dim=0) # [4b, num_classes]

    if soft_label: 
        labels_concat = preds # [4b, num_classes]
    else: 
        y_onehot = F.one_hot(y, num_classes=num_classes)
        labels_concat = torch.cat([y_onehot, y_onehot], dim=0) # [4b, num_classes]

    
    # option 1: total = ascl(cat[z1,z2], cat[z1a, z2a])
    if combine_type == 1:
        loss_ascl = soft_lcscl(hidden=torch.cat([nat_z, adv_z], dim=0).double(), 
                        labels=labels_concat.double(), 
                        preds=preds.double(), 
                        dist=dist, 
                        neg_type=neg_type, 
                        hidden_norm=hidden_norm, 
                        temperature=temperature)

    # ---------------------------------------------------------
    elif combine_type == 2: 
        # option 2: total = 0.5 * ascl(z1, z2) + 0.5 * ascl(z1a, z2a)
        p1, p2, p1a, p2a = torch.split(preds, batch_size, dim=0) # 4 pred labels, each shape: [b, num_classes]
        nat_z1, nat_z2 = torch.split(nat_z, batch_size, dim=0) # 2 nat-z, each shape [b, dim]
        adv_z1, adv_z2 = torch.split(adv_z, batch_size, dim=0) # 2 adv-z, each shape [b, dim]
        l1, l2, l1a, l2a = torch.split(labels_concat, batch_size, dim=0) # 4 true labels, each shape: [b, num_classes]

        loss_ascl = 0.5 * soft_lcscl(hidden=torch.cat([nat_z1, nat_z2], dim=0).double(), 
                            labels=torch.cat([l1, l2], dim=0).double(), 
                            preds=torch.cat([p1, p2], dim=0).double(), 
                            dist=dist, 
                            neg_type=neg_type, 
                            hidden_norm=hidden_norm, 
                            temperature=temperature)

        loss_ascl += 0.5 * soft_lcscl(hidden=torch.cat([adv_z1, adv_z2], dim=0).double(), 
                            labels=torch.cat([l1a, l2a], dim=0).double(), 
                            preds=torch.cat([p1a, p2a], dim=0).double(), 
                            dist=dist, 
                            neg_type=neg_type, 
                            hidden_norm=hidden_norm, 
                            temperature=temperature)
    # ---------------------------------------------------------

    elif combine_type == 3: 
        # option 3: total = 0.5 * ascl(z1, z1a) + 0.5 * ascl(z2, z2a)
        p1, p2, p1a, p2a = torch.split(preds, batch_size, dim=0) # 4 pred labels, each shape: [b, num_classes]
        nat_z1, nat_z2 = torch.split(nat_z, batch_size, dim=0) # 2 nat-z, each shape [b, dim]
        adv_z1, adv_z2 = torch.split(adv_z, batch_size, dim=0) # 2 adv-z, each shape [b, dim]
        l1, l2, l1a, l2a = torch.split(labels_concat, batch_size, dim=0) # 4 true labels, each shape: [b, num_classes]

        loss_ascl = 0.5 * soft_lcscl(hidden=torch.cat([nat_z1, adv_z1], dim=0).double(), 
                            labels=torch.cat([l1, l1a], dim=0).double(), 
                            preds=torch.cat([p1, p1a], dim=0).double(), 
                            dist=dist, 
                            neg_type=neg_type, 
                            hidden_norm=hidden_norm, 
                            temperature=temperature)

        loss_ascl += 0.5 * soft_lcscl(hidden=torch.cat([nat_z2, adv_z2], dim=0).double(), 
                            labels=torch.cat([l2, l2a], dim=0).double(), 
                            preds=torch.cat([p2, p2a], dim=0).double(), 
                            dist=dist, 
                            neg_type=neg_type, 
                            hidden_norm=hidden_norm, 
                            temperature=temperature)          

    loss_vat = kl_loss_with_logits(adv_output, nat_output, reduction='mean')
    loss_com = torch.mean(my_norm(adv_z, nat_z, p=dist, normalize=hidden_norm))

    loss = gbcomw*loss_ascl 
    loss += add_loss(loss_vat, lcsmtw)
    loss += add_loss(loss_com, lccomw)
    loss += add_loss(loss_natural, alpha)
    loss += add_loss(loss_robust, beta)

    log = dict()
    log['loss_nat_ce'] = loss_natural
    log['loss_adv_ce'] = loss_robust
    log['loss_ascl'] = loss_ascl
    log['loss_vat'] = loss_vat
    log['loss_com'] = loss_com

    return loss, x_adv, log