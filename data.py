import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

"""
Backward/forward-compatible data loader helper for this repo.

- Supports torchvision versions that provide dataset.train_labels / test_labels
  as well as newer versions that provide dataset.targets.
- Creates a validation split from the end of the training set (last 5000 examples)
  when use_test is False (this matches the original repo behavior).
- Returns a dict of DataLoaders and the number of classes.
"""

class Transforms:
    class VGG:
        @staticmethod
        def Compose():
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    class ResNet:
        @staticmethod
        def Compose():
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010]),
            ])


def _dataset_labels(dataset):
    """
    Return labels list for a torchvision dataset supporting different attribute names.
    """
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "train_labels"):
        labels = dataset.train_labels
    elif hasattr(dataset, "test_labels"):
        labels = dataset.test_labels
    else:
        # As a fallback try to extract labels from dataset samples
        try:
            labels = [t for (_, t) in dataset]
        except Exception:
            raise AttributeError("Unable to extract labels from dataset object.")
    # Convert tensors/numpy arrays to plain python list
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    try:
        # numpy arrays
        import numpy as _np
        if isinstance(labels, _np.ndarray):
            labels = labels.tolist()
    except Exception:
        pass
    return labels


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True):
    """
    Builds data loaders.

    Args:
      dataset (str): dataset name string recognizable by torchvision.datasets (e.g., 'CIFAR10' or 'CIFAR100')
      path (str): root path where data will be downloaded / located
      batch_size (int): batch size for DataLoaders
      num_workers (int): number of workers for DataLoaders
      transform_name (str): 'VGG' or 'ResNet' (selects normalization and augmentations)
      use_test (bool): if True evaluate on the official test set; otherwise use last 5000 training images as validation
      shuffle_train (bool): whether to shuffle the training DataLoader
    Returns:
      (dict_of_dataloaders, num_classes)
    """
    ds_class = getattr(torchvision.datasets, dataset)

    os.makedirs(path, exist_ok=True)

    # select transform
    transform_name = transform_name or "VGG"
    if transform_name.upper().startswith("RES"):
        transform = Transforms.ResNet.Compose()
    else:
        transform = Transforms.VGG.Compose()

    # full training set (used to create validation split if needed)
    train_full = ds_class(root=path, train=True, download=True, transform=transform)

    # determine number of classes from labels
    labels = _dataset_labels(train_full)
    if len(labels) == 0:
        raise RuntimeError("Empty training labels; check dataset path or download.")
    num_classes = int(max(labels)) + 1

    if use_test:
        # use official test set
        test_set = ds_class(root=path, train=False, download=True, transform=transform)
        train_set = train_full
    else:
        # create validation split from last 5000 train images
        n_total = len(train_full)
        n_val = 5000
        if n_total <= n_val:
            raise RuntimeError(f"Training set too small ({n_total}) to create validation split of {n_val}.")
        train_indices = list(range(0, n_total - n_val))
        val_indices = list(range(n_total - n_val, n_total))
        train_set = Subset(train_full, train_indices)
        test_set = Subset(train_full, val_indices)

    # DataLoaders: pin_memory True helps when using CUDA; for MPS it is ignored.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return {
        'train': train_loader,
        'test': test_loader,
    }, num_classes