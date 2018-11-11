from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset, ConcatDataset
from tqdm import tqdm
import numpy as np
import sys
from train import train, test
from model import Net
import matplotlib
from PIL import Image
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def collate_mixed_domain(data):
    data_x, data_y, data_domain = zip(*data)
    final_x = torch.stack(data_x)
    final_y = torch.from_numpy(np.array(data_y))
    final_domain = torch.from_numpy(np.array(data_domain, dtype=np.float32))

    return (final_x, final_y, final_domain)

class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)

class MixedDomainDataset(Dataset):
    def __init__(self):
        super(MixedDomainDataset)
        self.source_dataset = datasets.MNIST('./dataMNIST', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.target_dataset = datasets.SVHN('./dataSVHN', download=True,
                       transform=transforms.Compose([
                            transforms.Grayscale(),
                            transforms.RandomCrop(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)
                             )
                       ]))
    def __getitem__(self, index):
        domain_number = 0.0
        if index >= len(self.source_dataset):
            index = len(self.source_dataset) - index
            domain_number = 1.0
            return self.target_dataset[index][0], self.target_dataset[index][1], domain_number
        else:
            return self.source_dataset[index][0], self.source_dataset[index][1].item(), domain_number

    def __len__(self):
        return len(self.source_dataset) + len(self.target_dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = MixedDomainDataset()
    print("MixedDomainDataset: ", len(train_dataset))

    source_dataset = datasets.MNIST('./dataMNIST', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    target_dataset = datasets.SVHN('./dataSVHN', download=True,
                split = 'test',
                   transform=transforms.Compose([
                        transforms.Grayscale(),
                        transforms.RandomCrop(28),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)
                         )
                   ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_mixed_domain, **kwargs)
    test_loader_source = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader_target = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net(1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    epoch_train_loss = []
    epoch_train_accuracy = []
    print("Training on Task 1...")
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch,
                epoch_train_loss, epoch_train_accuracy)
        test(model, device, test_loader_source)
        test(model, device, test_loader_target)

    task1_train = epoch_train_accuracy
    task1_loss = epoch_train_loss
    epoch_train_accuracy = []
    epoch_train_loss = []


if __name__ == '__main__':
    main()
