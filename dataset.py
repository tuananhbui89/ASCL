import random
import numpy as np
import torch
from torchvision import datasets, transforms

def load_mnist_data():
    transform=transforms.Compose([
            transforms.ToTensor(),
            ])
    train_data = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    test_data = datasets.MNIST('../data', train=False,
                        transform=transform)

    return train_data, test_data


def load_cifar10_data():
    """
    Tried with two options: 
        - use standard RandomCrop(32, padding=4)
        - use RandomResizedCrop(32, scale=(0.2, 1.))
    Second approach has very worse performance compare with the standard one. 
    i.e., Performance with ResNet18, CIFAR10 
    Zhang-2020, PGD-AT, nat/pgd200: 80.15/43.46
    Zhang-2019, PGD-AT, nat/pgd200: 78.74/42.20
    Pang-2020, PGD-AT, nat/pgd200: 78.92/38.07
    Rice-2020, PGD-AT, nat/pgd200: 68.12/35.12 
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Standard  
        # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)), # Copy from SupContrast,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform_train)
    test_data = datasets.CIFAR10('../data', train=False,
                        transform=transform_test)    

    return train_data, test_data


def load_imagenet_test_data(test_batch_size=1, folder='../val/'):
    val_dataset = datasets.ImageFolder(
        folder,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    rand_seed = 42

    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=True)

    return val_loader

def load_data(ds='cifar10', train_batch_size=128, test_batch_size=128, folder='../val/'): 

    train_kwargs = {'batch_size': train_batch_size, 'shuffle': True, 
                    'num_workers': 1, 'pin_memory': True, 'drop_last': True} #'drop_last': True
    test_kwargs = {'batch_size': test_batch_size, 
                    'num_workers': 1, 'pin_memory': True}

    if ds == 'imagenet': 
        test_loader = load_imagenet_test_data(test_batch_size, folder)
        return None, test_loader

    elif ds == 'mnist': 
        train_data, test_data = load_mnist_data()

    elif ds == 'cifar10': 
        train_data, test_data =  load_cifar10_data()
    
    else: 
        raise ValueError
    
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)    

    return train_loader, test_loader

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def load_two_transforms(ds, train_batch_size=128, test_batch_size=128, folder='../val/'):
    """
    Note that 
    -   in this loader, we will not use normalizing in the end of the loader as in 
        original implementation. The normalizing job has been moved to buidling model job. 
        detail in `models.py` 
    -   for train loader, we will get two random batches 

    """
    assert(ds == 'cifar10')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Standard  
        # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)), # Copy from SupContrast,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    #     ], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor(),
    # ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if ds == 'cifar10':
        train_data = datasets.CIFAR10(root='../data',
                                         transform=TwoCropTransform(transform_train),
                                         download=True)
        test_data = datasets.CIFAR10('../data', train=False,
                            transform=transform_test)                                             
    elif ds == 'cifar100':
        train_data = datasets.CIFAR100(root='../data',
                                          transform=TwoCropTransform(transform_train),
                                          download=True)
        test_data = datasets.CIFAR100('../data', train=False,
                            transform=transform_test)                                               
    else:
        raise ValueError(ds)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True, sampler=train_sampler)
    
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    return train_loader, test_loader