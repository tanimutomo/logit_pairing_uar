import os
import numpy as np
import torch
import torchvision

from torchvision import datasets, transforms

from src.dataset.util import SubsetSampler


def load_dataset(dataset, batch_size, data_root, noise=False,
                 stddev=0.0, adv_subset=1000, workers=4):
    # mean and std
    mean = [0.49139968, 0.48215841, 0.44653091]
    std = [0.24703223, 0.24348513, 0.26158784]

    # path to dataset
    dataset_path = os.path.join(data_root, dataset)

    # train set
    train_dataset = datasets.CIFAR10(
        root=dataset_path,
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]))

    # validation set
    val_dataset = datasets.CIFAR10(
        root=dataset_path,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]))

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=workers)

    # val loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=workers)

    # adv val loader
    np.random.seed(123)  # load always the same random subset
    indices = np.random.choice(
        np.arange(val_dataset.__len__()),
        subset)
    subset_sampler = SubsetSampler(indices)

    aval_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, sampler=subset_sampler,
        num_workers=workers)


    return train_loader, val_loader, aval_loader

