import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from PIL import Image


def get_dataloader(name, train, args):
    """Return the dataloader for a given dataset."""
    batch_size = args.batch_size if train else args.test_batch_size
    if type(name) is not str:
        # MixedDomainDataset
        dataset = MixedDomainDataset(*name, train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_mixed_domain, **args.dataloader_kwargs)
    else:
        # Regular Dataset
        dataset = get_dataset(name, train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            **args.dataloader_kwargs)

    return loader


def get_dataset(name, train):
    """Get a dataset to be put into a dataloader.

    name: one of ['mnist', 'svhn']. These are the currently supported datasets.
    train: one of [True, False]. Which split of the data to use.
    """
    if name.lower() == 'mnist':
        return datasets.MNIST('./dataMNIST', train=train, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]))
    elif name.lower() == 'svhn':
        split = 'train' if train else 'test'
        return datasets.SVHN('./dataSVHN', download=True, split=split,
                             transform=transforms.Compose([
                                 transforms.Grayscale(),
                                 transforms.RandomCrop(28),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))]))
    else:
        raise ValueError(f"Dataset {name} not supported.")


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


def collate_mixed_domain(data):
    data_x, data_y, data_domain = zip(*data)
    final_x = torch.stack(data_x)
    final_y = torch.from_numpy(np.array(data_y))
    final_domain = torch.from_numpy(np.array(data_domain, dtype=np.float32))

    return (final_x, final_y, final_domain)


class MixedDomainDataset(Dataset):
    def __init__(self, source_dataset_name, target_dataset_name, train=True):
        super(MixedDomainDataset)
        self.source_dataset = get_dataset(source_dataset_name, train)
        self.target_dataset = get_dataset(target_dataset_name, train)

    def __getitem__(self, index):
        if index >= len(self.source_dataset):
            index = len(self.source_dataset) - index
            domain_number = 1.0
            return self.target_dataset[index][0], self.target_dataset[index][1], domain_number
        else:
            domain_number = 0.0
            return self.source_dataset[index][0], self.source_dataset[index][1].item(), domain_number

    def __len__(self):
        return len(self.source_dataset) + len(self.target_dataset)
