import torch.optim as optim
import argparse
import torch

from train import train, test
from model import Net

# Local imports
from utils import get_dataloader


def run_DA_model(source_dataset, target_dataset, args):
    """Function for running experiments using the domain adaptation model."""

    # Get the DataLoaders
    train_loader = get_dataloader([source_dataset, target_dataset], True, args)
    test_source_loader = get_dataloader(source_dataset, False, args)
    test_target_loader = get_dataloader(target_dataset, False, args)

    # Initialize our model
    model = Net(1).to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    epoch_train_loss = []
    epoch_train_accuracy = []
    print("Training on Task 1...")
    for epoch in range(1, args.epochs + 1):
        train(model, args.device, train_loader, optimizer, epoch,
              epoch_train_loss, epoch_train_accuracy)
        test(model, args.device, test_source_loader)
        test(model, args.device, test_target_loader)

    task1_train = epoch_train_accuracy
    task1_loss = epoch_train_loss
    epoch_train_accuracy = []
    epoch_train_loss = []


def main(args):
    # Set the device and random seed
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if use_cuda else "cpu")
    args.dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    run_DA_model('mnist', 'svhn', args)


if __name__ == '__main__':
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
    main(args)
