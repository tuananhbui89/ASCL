import torch 
import torch.functional as F 

def distance(x, y, dist, pairwise=False): 
    """
        Pairwise distance
        Args: 
            x, y: input pair 
            dist: distance type ["l2", "l1", "linf", "cosine"]
            pairwise: 
        Note: 
            Need renormalize if using l1
    """

    assert(len(x.shape) == 2)
    b, d = x.shape

    if pairwise:
        x = torch.unsqueeze(x, dim=0) # [1, b, d]
        y = torch.unsqueeze(y, dim=1) # [b, 1, d]

    if dist == 'l2':
        loss = torch.sqrt(torch.sum(torch.square(x - y), dim=-1, keepdim=False)) # [b,b]

    elif dist == 'l1': 
        loss = torch.sum(torch.abs(x - y), dim=-1, keepdim=False) # [b,b]

    elif dist == 'linf': 	
        loss, _ = torch.max(torch.abs(x - y), dim=-1, keepdim=False) # [b,b]

    elif dist == 'lkinf':
        k = 10 
        t, _ = torch.topk(torch.abs(x - y), k=k, dim=-1)
        loss = torch.sum(t, dim=-1, keepdim=False)
        loss /= k 

    elif dist == 'cosine': 
        # Note that: cosine_similarity range [-1,1]
        # dist=-1 <--> very similar 
        # dist=0 <--> orthogonal 
        # dist=1 <--> very disimilar
        loss = 0 - torch.cosine_similarity(x, y, dim=-1)

    elif dist == 'matmul': 
        x = torch.squeeze(x) # [b,d]
        y = torch.squeeze(y) # [b,d]
        loss = - torch.matmul(x, y.T)

    # loss = loss / torch.max(loss).detach() # normalizing to avoid numerical problem 

    if pairwise: 
        assert(loss.shape[0] == b)
        assert(loss.shape[1] == b)
    else:
        assert(loss.shape[0] == x.shape[0]) # [b, ]

    return loss 