from os import EX_SOFTWARE
import numpy as np 
import torch  
import torch.nn.functional as F 
from utils import one_hot_tensor

def convert2onehot(labels, num_classes=10): 
    if len(labels.shape) == 2: 
        return labels
    elif len(labels.shape) == 1: 
        return one_hot_tensor(labels, num_classes=num_classes)
        
def mysoftmax_cross_entropy_with_logits(logits, labels, reduction='mean'): 
    assert(len(logits.shape) == 2) 
    assert(len(labels.shape) == 2)
    t = -torch.mul(F.log_softmax(logits, dim=1), labels)
    t = torch.sum(t, dim=1)
    if reduction == 'mean': 
        return torch.mean(t, dim=0)
    elif reduction == 'sum': 
        return torch.sum(t, dim=0)
    elif reduction == 'none': 
        return t 
    else: 
        raise ValueError

def mysoftmax_cross_entropy_with_two_logits(logits, labels, reduction='mean'): 
    return mysoftmax_cross_entropy_with_logits(logits=logits, labels=F.softmax(labels), reduction=reduction)

def label_weight(y1, y2, scale, pairwise, one_hot=False, num_classes=10):
    # Scale should in (0., 1.)
    assert(y1.shape == y2.shape)
    if not one_hot: 
        assert(len(y1.shape) == 1)
        # print('** HARD CODE, label_weight, num_classes=10')
        y1 = one_hot_tensor(y1, num_classes=num_classes)
        y2 = one_hot_tensor(y2, num_classes=num_classes)
    
    if pairwise: 
        y1_ = torch.unsqueeze(y1, dim=0) # [1, b, 10]
        y2_ = torch.unsqueeze(y2, dim=1) # [b, 1, 10]
    else: 
        y1_ = y1 
        y2_ = y2 
    
    d = torch.abs(y1_ - y2_) # [b, b, 10] if pairwise else [b, 10]
    d = torch.sum(d, dim=-1) # [b, b] if pairwise else [b,]
    d = d / torch.max(d) # [0., 1.]
    d = scale -d # [scale, scale - 1]
    d = d / scale # [1, 1 - 1/scale]

    return d 

def compact_loss(z, z_p, pairwise): 
    # z, z_p shape [b, k]
    # z_p: latent vector of adversarial input 
    # z: latent vector of natural input 
    if pairwise: 
        z = torch.unsqueeze(z, dim=0) # [1, b, k]
        z_p = torch.unsqueeze(z_p, dim=1) # [b, 1, k]
    
    loss = torch.mean(torch.abs(z.detach() - z_p), dim=-1) # [b,b] if pairwise, else [b,]
    return loss 

def my_cross_entropy_with_two_logits(labels, logits): 
    # [b, b, d]
    # requireing inputs have 3 dimensions 
    assert(len(labels.shape) == 3)
    assert(len(logits.shape) == 3)
    b, _, d = labels.shape

    _labels = torch.reshape(labels, [b*b, d])
    _logits = torch.reshape(logits, [b*b, d])
    loss = mysoftmax_cross_entropy_with_two_logits(logits=_logits, labels=_labels, reduction='none') # [b*b, ]
    loss = torch.reshape(loss, [b, b])
    return loss 

def smooth_loss(logits, logits_p, pairwise): 
    # logits, logits_p shape [b, d]
    # logits_p: output of adversarial input 
    # logits: output of natural input 

    if pairwise: 
        b, d = logits.shape
        logits = torch.unsqueeze(logits, dim=0) # [1, b, d]
        logits_p = torch.unsqueeze(logits_p, dim=1) # [b, 1, d]

        logits = torch.repeat_interleave(logits, b, dim=0) # [b, b, d]
        logits_p = torch.repeat_interleave(logits_p, b, dim=1) # [b, b, d]

        loss = my_cross_entropy_with_two_logits(logits=logits_p, 
            labels=logits.detach())

        assert(logits.shape == logits_p.shape)
        assert(len(logits.shape) == 3)
        assert(logits.shape[0] == b)        
        assert(loss.shape[0] == b)
        assert(loss.shape[1] == b)
        
    else: 
        loss = mysoftmax_cross_entropy_with_two_logits(logits=logits_p, 
            labels=logits.detach(), reduction='none')

    return loss 

def global_loss(y, z, logits, y_p, z_p, logits_p, scale=0.99, num_classes=10): 
    # Global scale should be smaller than local scale 
    # y, z, logits: groundtruth, latent, output corresponding with natural input 
    # y_p, z_p, logits_p: groundtruth, latent, output corresponding with adversarial input 
    # scale: label scaling factor 
    assert(scale >= 0.9)
    c_loss = compact_loss(z, z_p, pairwise=True)
    s_loss = smooth_loss(logits, logits_p, pairwise=True)
    lw = label_weight(y1=y, y2=y_p, scale=scale, pairwise=True, one_hot=False, num_classes=num_classes)
    c_loss_weighted = torch.mean(torch.mul(lw.detach(), c_loss))
    s_loss_weighted = torch.mean(torch.mul(lw.detach(), s_loss))

    return c_loss_weighted, s_loss_weighted

def local_loss(y, z, logits, y_p, z_p, logits_p): 
    c_loss = compact_loss(z, z_p, pairwise=False)
    s_loss = smooth_loss(logits, logits_p, pairwise=False)
    return torch.mean(c_loss), torch.mean(s_loss)

def kl_loss_with_logits(logits_1, logits_2, reduction='mean', epsilon=1e-9): 
    assert(len(logits_1.shape)==2)
    assert(len(logits_2.shape)==2)

    # probs_1 = F.softmax(logits_1, dim=-1)
    # probs_2 = F.softmax(logits_2, dim=-1)

    # l = torch.sum(probs_1 * (torch.log(probs_1+epsilon) - torch.log(probs_2+epsilon)), dim=-1)

    # if reduction == 'none': 
    #     return l 
    # elif reduction == 'mean': 
    #     return torch.mean(l, dim=0)
    # elif reduction == 'sum': 
    #     return torch.sum(l, dim=0)

    return torch.nn.KLDivLoss(reduction=reduction)(F.log_softmax(logits_1, dim=-1), 
                                                F.softmax(logits_2, dim=-1))

def my_norm(x1, x2, p='l1', normalize=True, dim=-1):
    if normalize: 
        x1 = F.normalize(x1, 2, dim=-1)
        x2 = F.normalize(x2, 2, dim=-1)

    if p in ['l1', 1]: 
        return torch.norm(x1-x2, p=1, dim=dim)
    elif p in ['l2', 2]: 
        return torch.norm(x1-x2, p=2, dim=dim)
    elif p in ['linf', np.inf]: 
        return torch.norm(x1-x2, p=np.inf, dim=dim)
    elif p in ['cosine', 'matmul']: 
        return 1. - F.cosine_similarity(x1, x2, dim=dim)
