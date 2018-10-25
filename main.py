from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
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
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_task1, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn_task1, **kwargs)


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

	#Setting training as Task 2
	train_loader2 = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_task2, **kwargs)

	# Setting test as Task 1
	test_loader1 = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn_task1, **kwargs)

	test_loader2 = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn_task2, **kwargs)

	print("Training on Task 2...")
	for epoch in range(1, args.epochs + 1):
		train(model, device, train_loader2, optimizer, epoch, 
				epoch_train_loss, epoch_train_accuracy)
		print("Task 1:")
		test(model, device, test_loader1)
		print("Task 2:")
		test(model, device, test_loader2)

	task2_train = epoch_train_accuracy
	task2_loss = epoch_train_loss
	epoch_train_accuracy = []
	epoch_train_loss = []

	print("Retraining on Task 1...")
	for epoch in range(1, args.epochs + 1):
		train(model, device, train_loader, optimizer, epoch, 
				epoch_train_loss, epoch_train_accuracy)
		print("Task 1:")
		test(model, device, test_loader1)
		print("Task 2:")
		test(model, device, test_loader2)

	retrain_task1_train = epoch_train_accuracy
	retrain_task1_loss = epoch_train_loss
	epoch_train_accuracy = []
	epoch_train_loss = []

	all_train_accuracy = np.array([task1_train, task2_train, retrain_task1_train]).flatten()
	all_train_loss = np.array([task1_loss, task2_loss, retrain_task1_loss]).flatten()
	print(all_train_accuracy)
	np.save("train_acc", all_train_accuracy)
	np.save("train_loss", all_train_loss)
	plt.ylim(0,1)
	plt.plot(range(len(all_train_accuracy)), all_train_accuracy)
	plt.savefig("TrainingResults_MNIST.jpg")

	print("Evaluating on full dataset...")
	test_loader_all = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=False, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=args.test_batch_size, shuffle=True, **kwargs)
	test(model, device, test_loader_all)

if __name__ == '__main__':
	main()