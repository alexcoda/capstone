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
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def collate(data, classes):
	images, cl = zip(*data)
	images = torch.stack(images, 0)
	cl = torch.stack(cl, 0)
	final_images = []
	final_classes = []
	idx = []
	for i,c in enumerate(cl):   
		if c.data in classes:
			idx.append(i)

	final_images = images[idx,:,:,:]
	final_classes = cl[idx]

	return (final_images, final_classes)

def collate_fn_task1(data):
	classes = [0,1,2,3,4]
	return collate(data, classes)

def collate_fn_task2(data):
	classes = [5,6,7,8,9]
	return collate(data, classes)

def collate_mixed_domain(data):
	data_xy, data_domain = zip(*data)
	data_x, data_y = zip(*data_xy)
	final_x = torch.stack(data_x)
	final_y = torch.stack(data_y)
	final_domain = torch.stack(data_domain)

	return (final_x, final_y, final_domain)

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
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))
	def __getitem__(self, index):
		#TODO
		domain_number = 0
		if index > len(self.source_dataset):
			index = len(self.source_dataset) - index
			domain_number = 1
			# print(self.target_dataset[index])
			return self.target_dataset[index], domain_number
		else:
			return self.source_dataset[index], domain_number

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

	mnistDataset = datasets.MNIST('./dataMNIST', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))
	svhnDataset = datasets.SVHN('./dataSVHN', download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))
	train_dataset = MixedDomainDataset()
	print("MixedDomainDataset: ", len(train_dataset))
	# train_source_loader = torch.utils.data.DataLoader(
	# 	datasets.MNIST('./dataMNIST', train=True, download=True,
	# 				   transform=transforms.Compose([
	# 					   transforms.ToTensor(),
	# 					   transforms.Normalize((0.1307,), (0.3081,))
	# 				   ])),
	# 	batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_task1, **kwargs)
	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size, shuffle=True, collate_fn=collate_mixed_domain, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_mixed_domain, **kwargs)


	model = Net().to(device)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

	epoch_train_loss = []
	epoch_train_accuracy = []
	print("Training on Task 1...")
	for epoch in range(1, args.epochs + 1):
		train(model, device, train_loader, optimizer, epoch, 
				epoch_train_loss, epoch_train_accuracy)
		test(model, device, test_loader)

	task1_train = epoch_train_accuracy
	task1_loss = epoch_train_loss
	epoch_train_accuracy = []
	epoch_train_loss = []


if __name__ == '__main__':
	main()