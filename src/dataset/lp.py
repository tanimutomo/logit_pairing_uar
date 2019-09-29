import os
import numpy as np
import torch
import torchvision

from  torchvision import transforms
from torch.utils.data.sampler import Sampler 

from src.dataset.util import SubsetSampler


def load_dataset(dataset, batch_size, data_root, noise=False,
                 stddev=0.0, adv_subset=1000, workers=4):
    data_dir = os.path.join(data_root, dataset)
    if dataset == 'mnist':
        train_loader, test_loader, adv_test_loader, input_shape, num_classes = \
                load_mnist(batch_size, data_dir, noise,
                           stddev, adv_subset, workers)
    elif dataset == 'cifar10':
        train_loader, test_loader, adv_test_loader, input_shape, num_classes = \
                load_cifar10(batch_size, data_dir, noise,
                             stddev, adv_subset, workers)
    else:
        raise NotImplementedError

    return train_loader, test_loader, adv_test_loader, input_shape, num_classes
        

def load_mnist(batch_size, data_dir, noise=False, stddev=0.0, adv_subset=1000, workers=4):
    trainloader, _, classes = get_mnist(batch_size=batch_size,
                                        train=True,
                                        path=data_dir,
                                        noise=noise,
                                        std=stddev,
                                        shuffle=True,
                                        workers=workers
                                        )

    testloader, _, _ = get_mnist(batch_size=batch_size,
                                 train=False,
                                 path=data_dir,
                                 shuffle=False,
                                 workers=workers
                                 )

    adv_testloader, _, _ = get_mnist(batch_size=batch_size,
                                     train=False,
                                     path=data_dir,
                                     shuffle=False,
                                     adversarial=True,
                                     subset=adv_subset,
                                     workers=workers
                                     )

    input_shape = (None, 28, 28, 1)

    return trainloader, testloader, adv_testloader, input_shape, len(classes)


def load_cifar10(batch_size, data_dir, noise=False, stddev=0.0, adv_subset=1000, workers=4):

    trainloader, _, classes = get_cifar10(batch_size=batch_size,
                                          train=True,
                                          path=data_dir,
                                          noise=noise,
                                          std=stddev,
                                          shuffle=True,
                                          workers=workers
                                          )

    testloader, _, _ = get_cifar10(batch_size=batch_size,
                                   train=False,
                                   path=data_dir,
                                   shuffle=False,
                                   workers=workers
                                   )

    adv_testloader, _, _ = get_cifar10(batch_size=batch_size,
                                       train=False,
                                       path=data_dir,
                                       shuffle=False,
                                       adversarial=True,
                                       subset=adv_subset,
                                       workers=workers
                                       )

    input_shape = (None, 32, 32, 3)

    return trainloader, testloader, adv_testloader, input_shape, len(classes)


def get_mnist(batch_size, train, path, noise=False, std=0.0, shuffle=True, adversarial=False, subset=1000,
              workers=0):
    classes = np.arange(0, 10)

    if noise:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.Tensor(x.size()).normal_(mean=0.0, std=std)),  # add gaussian noise
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

    dataset = torchvision.datasets.MNIST(root=path,
                                         train=train,
                                         download=True,
                                         transform=transform)

    if adversarial:
        np.random.seed(123)  # load always the same random subset
        indices = np.random.choice(np.arange(dataset.__len__()), subset)

        subset_sampler = SubsetSampler(indices)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 sampler=subset_sampler,
                                                 num_workers=workers
                                                 )

    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=workers
                                                 )
    return dataloader, dataset, classes


def get_cifar10(batch_size, train, path, noise=False, std=0.0, shuffle=True, adversarial=False, subset=1000,
                workers=0):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if noise:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.Tensor(x.size()).normal_(0.0, std)),  # add gaussian noise
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

    dataset = torchvision.datasets.CIFAR10(root=path,
                                           train=train,
                                           download=True,
                                           transform=transform)

    if adversarial:
        np.random.seed(123)  # load always the same random subset
        indices = np.random.choice(np.arange(dataset.__len__()), subset)

        subset_sampler = SubsetSampler(indices)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 sampler=subset_sampler,
                                                 num_workers=workers
                                                 )
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=workers
                                                 )
    return dataloader, dataset, classes

