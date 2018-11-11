import torch.nn.functional as F
import torch.optim as optim
import torch
import sys
import pdb

from model import LWFNet
from tqdm import tqdm


def train_LWF(train_source_loader, train_target_loader,
              test_target_loader, test_source_loader, args):
    model = LWFNet(args.lambd).to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    epoch_train_loss = []
    epoch_train_accuracy = []
    print("Training on Task 1...")
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, args.device, train_source_loader,
                    optimizer, epoch, epoch_train_loss, epoch_train_accuracy,
                    task_n=1)
        test(model, args.device, test_source_loader)
        test(model, args.device, test_target_loader)

    print("Training on Task 2...")
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, args.device, train_target_loader,
                    optimizer, epoch, epoch_train_loss, epoch_train_accuracy,
                    task_n=2)
        test(model, args.device, test_source_loader)
        test(model, args.device, test_target_loader)

    task1_train = epoch_train_accuracy
    task1_loss = epoch_train_loss
    epoch_train_accuracy = []
    epoch_train_loss = []


def multinomial_log_loss(preds, targets):
    """Compute the multinomial log loss for a prediction."""
    loss = - 1


def train_epoch(model, device, train_loader, optimizer, epoch,
                epoch_train_loss, epoch_train_accuracy, task_n):
    # Stats to track
    total_loss = 0
    batches = 0
    accuracy = 0
    n_examples = 0
    correct = 0

    model.train()

    # Wrap our data in a tqdm progress bar and iterate through all batches
    task_idx = task_n - 1
    pbar = tqdm(train_loader, file=sys.stdout)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        batches += 1

        # Predict on the data
        optimizer.zero_grad()
        output_classes = model(data)
        primary_task_classes = output_classes[task_idx]

        # Check if the max-log prediction was correct
        pred = primary_task_classes.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        n_examples += target.shape[0]

        # Get the total loss
        # if epoch > 2:
        pdb.set_trace()
        primary_loss = F.nll_loss(primary_task_classes, target)
        if task_n == 1:
            loss = primary_loss
        if task_n > 1:
            # Add in the loss for previously learned tasks
            # TODO
            # loss_domain = F.binary_cross_entropy_with_logits(output_domain, domain.view(-1, 1))
            prior_task_loss = 0  # TODO
            loss = primary_loss + prior_task_loss

        # Update the progress bar with this epoch's loss
        pbar.set_description(f"Loss: {loss.item():0.4f}")
        pbar.update(1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_train_loss.append(total_loss / batches)
    epoch_train_accuracy.append(correct / (n_examples * 1.0))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    n_examples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_class = model(data)[0]
            # Removed kwarg: reduction = 'sum' b/c not compliant with pytorch 4.0
            test_loss += F.nll_loss(output_class, target).item()
            # get the index of the max log-probability
            pred = output_class.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            n_examples += target.shape[0]

    test_loss /= n_examples
    perc_correct = 100. * correct / n_examples
    print(f"\nTest set: Average loss: {test_loss:0.4f},"
          f"Accuracy: {correct}/{n_examples} ({perc_correct:.0f}%)\n")
