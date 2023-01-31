import torch
from torch._C import device 
import torch.nn.functional as F

from distance import distance

def get_positive_mask(labels): 
    # ATTENTION HERE, positive mask will ignore the diagonal

    # l = torch.matmul(labels, labels.T) # [2b,2b]
    # m = torch.ones_like(l).fill_diagonal_(0) # [2b,2b]
    # pos_mask = torch.multiply(l, m) # [2b,2b]

    l = torch.matmul(labels, labels.T).fill_diagonal_(0) # [2b,2b]
    return l

def get_negative_mask(labels): 
    pos_mask = torch.matmul(labels, labels.T) # [2b, 2b], mask for y_j == y_i
    inv_mask = torch.ones_like(pos_mask) - pos_mask # [2b, 2b], mask for y_j != y_i
    return inv_mask

def get_hard_negative_mask(labels, preds): 
    pos_mask = torch.matmul(labels, labels.T) # [2b, 2b], mask for y_j == y_i
    inv_mask = torch.ones_like(pos_mask) - pos_mask # [2b, 2b], mask for y_j != y_i
    hard_mask = torch.matmul(labels, preds.T) # [2b, 2b], mask for tile(y_j) == y_i
    neg_mask = torch.multiply(inv_mask, hard_mask) # [2b, 2b], mask for (y_j!= y_i & tile(y_j) == y_i) 
    return neg_mask


def get_soft_negative_mask(labels, preds):
    pos_mask = torch.matmul(labels, labels.T) # [2b, 2b], mask for y_j == y_i
    inv_mask = torch.ones_like(pos_mask) - pos_mask # [2b, 2b], mask for y_j != y_i
    soft_mask = torch.matmul(preds, preds.T) # [2b, 2b], mask for tile(y_j) == tile(y_i)
    neg_mask = torch.multiply(inv_mask, soft_mask) # [2b, 2b], mask for (y_j!= y_i & tile(y_j) == y_i)  
    return neg_mask

LARGE_NUM = 1e9 

def soft_lcscl(hidden,
             labels,
             preds,
             dist,
             neg_type,
             hidden_norm=True,
             temperature=0.07):
    """Compute Local Supervised Contrastive Loss.
    Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        labels: labels of hidden vector of shape (2*bsz, classes)
        preds: current prediction 
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        dist: distance metric in latent space {'l1', 'l2', 'cosine', 'linf'}
        neg_type: 
            'hard': negative set when y_j ~= y_i, but tilde(y_j) == y_i 
            'soft': negative set when y_j ~= y_i, but tilde(y_j) == tilde(y_i)
            'merge': negative set when y_i ~= y_i but tilde(y_j) in [y_i,  tilde(y_i)]
            'leaking': 
    Returns:
        A loss scalar.

    the choice of temperature:
        temperature = 0.5 based on simCLR paper  
        temperature = 0.07 based on SCL paper (they claim that smaller temperature 
        benefit training more than higher ones)
    """
    if dist in ['matmul', 'l1', 'l2']: 
        assert(hidden_norm is True) 
    elif dist in ['cosine']: 
        assert(hidden_norm is False)
    
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        # gradient will pass through the normalization step 
        hidden = F.normalize(hidden, p=2, dim=-1)
    
    
    batch_size = hidden.shape[0] // 2
    
    # Remove illegal elements 
    same_preds = get_positive_mask(preds)
    same_labels = get_positive_mask(labels)
    diff_labels = get_negative_mask(labels)

    same_preds_same_labels = torch.multiply(same_preds, same_labels)
    same_preds_diff_labels = torch.multiply(same_preds, diff_labels)

    pos_mask = get_positive_mask(labels)
    # neg_mask = tf.ones_like(pos_mask) - pos_mask # INCORRECT, because it consider the diagonal as NEGATIVE 
    neg_mask = get_negative_mask(labels) # CORRECT 
    hard_neg_mask = get_hard_negative_mask(labels, preds)
    soft_neg_mask = get_soft_negative_mask(labels, preds)
    merge_mask = torch.minimum(hard_neg_mask+soft_neg_mask, torch.ones_like(hard_neg_mask))
    
    # ATTENTION HERE, NEED VERIFIED BY EXPERIMENT, Stop_gradient make it worse 
    # pos_mask = pos_mask.detach()
    # neg_mask = neg_mask.detach()
    # hard_neg_mask = hard_neg_mask.detach()
    # soft_neg_mask = soft_neg_mask.detach()
    # merge_mask = merge_mask.detach()
    # same_preds_same_labels = same_preds_same_labels.detach()
    #     
    if neg_type == 'all': 
        illegal_mask = torch.zeros_like(neg_mask)
        nb_neg = torch.sum(neg_mask, dim=-1)

    elif neg_type == 'hard': 
        illegal_mask = torch.maximum(neg_mask - hard_neg_mask, torch.zeros_like(neg_mask))
        nb_neg = torch.sum(hard_neg_mask, dim=-1)

    elif neg_type == 'soft': 
        illegal_mask = torch.maximum(neg_mask - soft_neg_mask, torch.zeros_like(neg_mask))
        nb_neg = torch.sum(soft_neg_mask, dim=-1)

    elif neg_type == 'merge': 
        illegal_mask = torch.maximum(neg_mask - merge_mask, torch.zeros_like(neg_mask))
        nb_neg = torch.sum(merge_mask, dim=-1)

    elif neg_type == 'leaking': 
        illegal_mask = torch.maximum(torch.ones_like(same_preds_same_labels) - same_preds_same_labels - same_preds_diff_labels, torch.zeros_like(same_preds_same_labels))
        nb_neg = torch.sum(same_preds_diff_labels, dim=-1)

    # Pairwise Similarity 
    masks = F.one_hot(torch.arange(2*batch_size), num_classes=2*batch_size).to(hidden.device)
    sim = - distance(hidden, hidden, dist, pairwise=True) / temperature
    logits_12 = sim - masks * LARGE_NUM
    logits_masked = logits_12 - LARGE_NUM * illegal_mask 
    logprobs = F.log_softmax(logits_masked, dim=1)

    """ Convert labels to pairwise labels of shape [2b, 2b], diagonal(labels)=0
        multiple hot label vectors 

    """ 
    # Cross entropy loss with multiple hot labels 

    if neg_type == 'leaking':
        loss = - torch.sum(same_preds_same_labels * logprobs, dim=-1) # [2b, ]
        nb_pos = torch.maximum(torch.sum(same_preds_same_labels, dim=-1), torch.ones_like(loss)) # [2b, ]
    else:
        loss = - torch.sum(pos_mask * logprobs, dim=-1) # [2b, ]
        nb_pos = torch.maximum(torch.sum(pos_mask, dim=-1), torch.ones_like(loss)) # [2b, ]

    # if True: 
    #     scale = torch.div(nb_neg, nb_neg+nb_pos) # V1
    #     # scale = torch.div(nb_neg, nb_pos) #V3
    #     # scale = nb_neg #V2
    #     loss = torch.multiply(loss, scale)
        
    assert(loss.shape == nb_pos.shape)
    loss = torch.div(loss, nb_pos) # [2b,]
    loss = torch.mean(loss, dim=0) # [1,]

    return loss 