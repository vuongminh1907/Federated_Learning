import copy
import torch
from torchvision import datasets, transforms
import numpy as np
from torchvision import datasets, transforms

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    
    user_groups = mnist_iid(train_dataset, args.num_users)
    
    return train_dataset, test_dataset, user_groups

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


    
    
   

    