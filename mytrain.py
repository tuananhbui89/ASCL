from math import log
from ascl import ascl_pgd
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.optim as optim
from functools import partial

from trades import trades_loss
from pgd import pgd_loss, pgd_attack, pgd_attack_old
from adr import adr_pgd, adr_trades
from ascl import ascl_pgd, ascl_trades
from ascl import scl_pgd
from ascl import ascl_pgd_2trans, ascl_trades_2trans
from ascl import scl_nat_2trans
from utils import count_pred 

def get_diff(X, X_adv, order, epsilon=None): 
    X = torch.reshape(X, [X.shape[0], -1])
    X_adv = torch.reshape(X_adv, [X_adv.shape[0], -1])
    d = torch.norm(X_adv - X, p=order, dim=-1, keepdim=True) # [b,]
    if epsilon is not None: 
        delta = torch.abs(X_adv - X) 
        num_exceed = torch.sum(delta > epsilon, dim=1)/X.shape[1]
        num_exceed = torch.mean(num_exceed)
        d = torch.mean(d) # []
        return d, num_exceed
    else: 
        d = torch.mean(d) # []
        return d 

def train(model, data_loader, epoch, optimizer, device, log_interval, attack_params, writer): 
    model.train()
    num_batches = len(data_loader.dataset) // 128

    for batch_idx, (data, target) in enumerate(data_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        nat_acc = get_acc(output, target)

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

    return writer

def adv_train(model, data_loader, epoch, optimizer, device, log_interval, attack_params, writer): 
    model.train()

    kwargs = {
        'step_size': attack_params['step_size'], 
        'epsilon': attack_params['epsilon'], 
        'perturb_steps': attack_params['num_steps'], 
        'alpha': attack_params['alpha'], 
        'beta': attack_params['trades_beta'], 
        'projecting': attack_params['projecting'], 
        'x_min': attack_params['x_min'], 
        'x_max': attack_params['x_max'],
    }

    kwargs_2 = {
        'lccomw': attack_params['lccomw'], 
        'lcsmtw': attack_params['lcsmtw'], 
        'gbcomw': attack_params['gbcomw'], 
        'gbsmtw': attack_params['gbsmtw'], 
        'confw': attack_params['confw'], 
    }

    kwargs_3 = {
        'dist': attack_params['dist'], 
        'neg_type': attack_params['neg_type'],
        'temperature': attack_params['tau'],
        'hidden_norm': attack_params['hidden_norm'],
        'combine_type': attack_params['combine_type'],
    }

    if attack_params['defense'] == 'pgd_train': 
        defense = pgd_loss 
    elif attack_params['defense'] == 'trades_train': 
        defense = trades_loss
    elif attack_params['defense'] == 'adr_pgd': 
        defense = adr_pgd
    elif attack_params['defense'] == 'adr_trades': 
        defense = adr_trades
    elif attack_params['defense'] == 'ascl_pgd': 
        defense = ascl_pgd
    elif attack_params['defense'] == 'ascl_trades': 
        defense = ascl_trades
    elif attack_params['defense'] == 'scl_pgd': 
        defense = scl_pgd
    elif attack_params['defense'] == 'scl_nat_2trans': 
        defense = scl_nat_2trans 
    elif attack_params['defense'] == 'ascl_pgd_2trans': 
        defense = ascl_pgd_2trans
    elif attack_params['defense'] == 'ascl_trades_2trans': 
        defense = ascl_trades_2trans
    else:
        raise ValueError 
    
    if attack_params['defense'] in ['adr_pgd', 'adr_trades']: 
        kwargs.update(kwargs_2)
    elif attack_params['defense'] in ['ascl_pgd', 'ascl_trades', 'scl_pgd', 'scl_nat_2trans', 'ascl_pgd_2trans', 'ascl_trades_2trans']:
        kwargs.update(kwargs_2)
        kwargs.update(kwargs_3)   

    if attack_params['defense'] in ['scl_nat_2trans', 'ascl_pgd_2trans', 'ascl_trades_2trans']: 
        twotrans = True 
    else: 
        twotrans = False 
    
    num_batches = len(data_loader.dataset) // 128

    for batch_idx, (data, target) in enumerate(data_loader):
        assert(model.training is True)
        if twotrans:
            data = torch.cat([data[0], data[1]], dim=0) # data shape [2b, C, H, W]
            target = torch.cat([target, target], dim=0) # target shape [2b,]
            assert(data.shape[0] == target.shape[0])

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        loss, X_adv, logloss = defense(model=model,
                           x_natural=data,
                           y=target,
                           device=device,
                           optimizer=optimizer,
                           **kwargs)

        model.train()
        loss.backward()
        optimizer.step()

        nat_output = model(data)
        adv_output = model(X_adv)
        nat_acc = get_acc(nat_output, target)
        adv_acc = get_acc(adv_output, target)

        if batch_idx % log_interval == 0:

            writestr = [
                ('Train_iter={}', epoch*num_batches + batch_idx),
                ('nat_acc={:.4f}', nat_acc.item()), 
                ('adv_acc={:.4f}', adv_acc.item()), 
                ('loss={:.4f}', loss.item()), 
            ]
            for key in logloss.keys(): 
                writestr.append((key+'={:.4f}', logloss[key].item()))

            writestr = '  ,'.join([t.format(v) for (t, v) in writestr]) 
            print(writestr)
            writer.add_scalar('nat_acc', nat_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('adv_acc', adv_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('loss', loss.item(), epoch*num_batches + batch_idx)      
            for key in logloss.keys(): 
                writer.add_scalar(key, logloss[key].item(), epoch*num_batches + batch_idx)

            for key in logloss.keys(): 
                if logloss[key].item() == np.nan: 
                    return writer

    return writer


def test(model, data_loader, device, return_count=False, num_classes=10): 
    model.eval()
    test_loss = 0
    correct = 0

    pred_as_count = np.zeros(shape=[num_classes,])
    correct_count = np.zeros(shape=[num_classes,])
    class_count = np.zeros(shape=[num_classes,])

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            p, c = count_pred(labels=target, preds=output, num_classes=num_classes)
            pred_as_count += p 
            correct_count += c 
            class_count += count_pred(labels=target, preds=target, num_classes=num_classes)[0]

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    print('pred_as_count: ', pred_as_count)
    print('correct_count: ', correct_count)
    
    if return_count: 
        return accuracy, pred_as_count, correct_count, class_count
    else: 
        return accuracy

def adv_test(model, data_loader, device, attack_params, return_count=False, num_classes=10): 
    model.eval()
    test_loss = 0
    correct = 0

    pred_as_count = np.zeros(shape=[num_classes,])
    correct_count = np.zeros(shape=[num_classes,])
    class_count = np.zeros(shape=[num_classes,]) 

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            X_adv, _ = pgd_attack_old(model, data, target, device, attack_params, status='eval')
            X_adv = Variable(X_adv.data, requires_grad=False)

            output = model(X_adv)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            p, c = count_pred(labels=target, preds=output, num_classes=num_classes)
            pred_as_count += p 
            correct_count += c 
            class_count += count_pred(labels=target, preds=target, num_classes=num_classes)[0]

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print('\nRobustness evaluation : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    
    if return_count: 
        return accuracy, pred_as_count, correct_count, class_count
    else: 
        return accuracy

def get_pred(model, data_loader, device): 
    model.eval()
    result = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            output = torch.nn.Softmax()(output)
            result.append(output.cpu().numpy())

    result = np.concatenate(result, axis=0)
    return result

def get_acc(output, target): 
    pred = output.argmax(dim=1, keepdim=True)
    acc = torch.mean(pred.eq(target.view_as(pred)).type(torch.FloatTensor))
    return acc 
