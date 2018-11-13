import torch.nn.functional as F
import torch.optim as optim
import torch
import copy
import sys

from model import LWFNet
from tqdm import tqdm


def train_LWF(train_source_loader, train_target_loader,
              test_target_loader, test_source_loader, args):

    epoch_train_loss = []
    epoch_train_accuracy = []

    """Initial training on task 1."""

    model = LWFNet(args.lambd).to(args.device)
    optim_kwargs = {'lr': args.lr,
                    'momentum': args.momentum,
                    'weight_decay': 0.0005}
    optimizer = optim.SGD(model.parameters(), **optim_kwargs)

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, args.device, train_source_loader,
                    optimizer, epoch, epoch_train_loss, epoch_train_accuracy,
                    task_n=1)
        test(model, args.device, test_source_loader, task_n=1)
        # task_n = 1 here because we only have 1 output layer at this point
        test(model, args.device, test_target_loader, task_n=1)

    """Warm-up step for task 2."""
    print("~~~ Warm up ~~~")

    orig_model = copy.deepcopy(model)
    model.add_prediction_layer(args.device, 10)
    optimizer = optim.SGD(model.parameters(), **optim_kwargs)
    freeze_parameters(model, task_n=2)

    for epoch in range(1, args.epochs + 1):

        train_epoch(model, args.device, train_target_loader,
                    optimizer, epoch, epoch_train_loss, epoch_train_accuracy,
                    task_n=2, phase='warm-up')
        test(model, args.device, test_source_loader, task_n=1)
        test(model, args.device, test_target_loader, task_n=2)

    """Fine-tune step for task 2."""
    print("~~~ Fine Tune ~~~")

    unfreeze_parameters(model, task_n=2)
    optim_kwargs['lr'] = optim_kwargs['lr'] / 10
    optimizer = optim.SGD(model.parameters(), **optim_kwargs)

    for epoch in range(1, args.epochs + 1):

        train_epoch(model, args.device, train_target_loader,
                    optimizer, epoch, epoch_train_loss, epoch_train_accuracy,
                    task_n=2, phase='fine-tune', orig_model=orig_model)
        test(model, args.device, test_source_loader, task_n=1)
        test(model, args.device, test_target_loader, task_n=2)

    task1_train = epoch_train_accuracy
    task1_loss = epoch_train_loss
    epoch_train_accuracy = []
    epoch_train_loss = []


# def multinomial_log_loss(preds, targets):
#     """Compute the multinomial log loss for a prediction."""
#     loss = - 1


def train_epoch(model, device, train_loader, optimizer, epoch,
                epoch_train_loss, epoch_train_accuracy, task_n,
                phase=None, orig_model=None):
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
        outputs = model(data)

        loss = get_loss(outputs, target, task_n, phase, data, orig_model)

        # Update the progress bar with this epoch's loss
        pbar.set_description(f"Loss: {loss.item():0.4f}")
        pbar.update(1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Check if the max-log prediction was correct
        primary_task_classes = outputs[task_idx]
        pred = primary_task_classes.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        n_examples += target.shape[0]

    epoch_train_loss.append(total_loss / batches)
    epoch_train_accuracy.append(correct / (n_examples * 1.0))

    print(f"Epoch Train Loss: {epoch_train_loss[-1]:0.4f}")
    print(f"Epoch Train Acc: {epoch_train_accuracy[-1]:0.4f}")


def get_loss(outputs, target, task_n, phase, data, orig_model):

    if task_n == 1:
        # Only get the loss from task 1
        loss = F.nll_loss(outputs[0], target)
    elif task_n > 1:
        if phase == 'warm-up':
            # Only get the loss from task 2
            loss = F.nll_loss(outputs[1], target)
        elif phase == 'fine-tune':
            orig_task1_output = orig_model(data)[0]
            task1_loss = 0.1 * F.mse_loss(outputs[0], orig_task1_output)
            task2_loss = F.nll_loss(outputs[1], target)
            loss = task1_loss + task2_loss
        else:
            raise ValueError(f"phase {phase} not supported.")

    return loss


def freeze_parameters(model, task_n):
    toggle_parameters(model, task_n, False)


def unfreeze_parameters(model, task_n):
    toggle_parameters(model, task_n, True)


def toggle_parameters(model, task_n, freeze):
    for param in model.feature_extractor.parameters():
        param.requires_grad = freeze
    for i, layer in enumerate(model.label_predictors):
        if (i + 1) != task_n:
            for param in layer.parameters():
                param.requires_grad = freeze


def test(model, device, test_loader, task_n):
    task_idx = task_n - 1

    model.eval()
    test_loss = 0
    correct = 0
    n_examples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_class = model(data)[task_idx]
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
