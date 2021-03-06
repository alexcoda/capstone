import argparse
import torch

# Local imports
from utils import get_dataloader, save_results
from train_LWF import train_LWF
from train_DA import train_DA


def run_DA_model(source_name, target_name, args):
    """Function for running experiments using the domain adaptation model."""

    # Get the DataLoaders
    train_loader = get_dataloader([source_name, target_name], True, args)
    test_source_loader = get_dataloader(source_name, False, args)
    test_target_loader = get_dataloader(target_name, False, args)

    # Train the model
    results_df = train_DA(train_loader, test_source_loader, test_target_loader, args)

    save_name = f"{source_name}-{target_name}_epoch={args.epochs}_lr={args.lr}_lambda={args.lambd}"
    save_results('da', save_name, results_df)


def run_LWF_model(source_name, target_name, args):
    """Function for running experiments using the learning without forgetting model."""

    # Get the DataLoaders
    train_source_loader = get_dataloader(source_name, True, args)
    train_target_loader = get_dataloader(target_name, True, args)
    test_source_loader = get_dataloader(source_name, False, args)
    test_target_loader = get_dataloader(target_name, False, args)

    # Train the model
    results_df = train_LWF(train_source_loader, train_target_loader,
                           test_target_loader, test_source_loader, args)

    save_name = f"{source_name}-{target_name}_epoch={args.epochs}_lr={args.lr}_lambda={args.lambd}"
    save_results('lwf', save_name, results_df)


def main(args):
    # Set the device and random seed
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if use_cuda else "cpu")
    args.dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    run_DA_model('mnist', 'svhn', args)

    # run_LWF_model('mnist', 'svhn', args)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Image classifier args')
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
    parser.add_argument('--lambd', type=float, default=0.1, metavar='L',
                        help='The parameter to balance remebering old data vs. learning new')
    args = parser.parse_args()
    main(args)
