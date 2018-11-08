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

def train(model, device, train_loader, optimizer, epoch, 
        epoch_train_loss, epoch_train_accuracy):
    global log_interval
    model.train()
    pbar = tqdm(train_loader, file=sys.stdout)
    total_loss = 0
    batches = 0
    accuracy = 0
    n_examples = 0
    correct =0 
    for (data, target, domain) in pbar:
        batches += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_class, output_domain = model(data)
        pred = output_class.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n_examples += target.shape[0]
        loss_label = F.nll_loss(output_class, target)
        loss_domain = F.binary_cross_entropy_with_logits()
        pbar.set_description("Loss: %.4f"%loss_label.item())
        pbar.update(1)
        loss_class.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_train_loss.append(total_loss/batches)
    epoch_train_accuracy.append(correct/(n_examples*1.0))
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    n_examples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n_examples += target.shape[0]

    test_loss /= n_examples
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, n_examples,
        100. * correct / n_examples))