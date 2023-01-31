import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

"""
We follow the finding in Bag of Trick paper (Pang et al. 2020) to made some change as follow: 
- We init PGD attack with uniform noise (-epsilon, epsilon)
- In the original implementation, the model has been change to evaluation stage (to change BN to evaluation). 
In this version, we do not change the model stage in the pgd attack. 
It means that: 
    - When training, the pgd attack will be generated with model in training stage 
    - When evaluation, the pgd attack will be generated with model in evaluation stage  
"""

def trades_loss(model,
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
    # define KL-loss
    assert(projecting is True)
    assert(distance == 'l_inf')
    assert(x_max > x_min)

    criterion_kl = nn.KLDivLoss(size_average=False)
    
    # model.eval()
    model.train()
    assert(model.training is True)

    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            if projecting:
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, x_min, x_max)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(x_min, x_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, x_min, x_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = alpha * loss_natural + beta * loss_robust
    log = dict()
    log['loss_nat_ce'] = loss_natural
    log['loss_adv_ce'] = loss_robust

    return loss, x_adv, log 
