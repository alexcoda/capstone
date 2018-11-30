import torch.nn.functional as F
import torch.optim as optim
import torch
import copy
import sys

from model import LWFNet
from tqdm import tqdm
from abc import ABC, abstractmethod


def train_LWF(train_source_loader, train_target_loader,
              test_target_loader, test_source_loader, args):

    # Initialize the model
    model = LWFNet().to(args.device)
    optim_kwargs = {'lr': args.lr,
                    'momentum': args.momentum,
                    'weight_decay': 0.0005}

    # Set-up the tests to run on our model
    source_tester = Tester(test_source_loader, args, task_id=0)
    target_tester = Tester(test_target_loader, args, task_id=1)

    """Initial training on task 1."""
    print("~~~ Training Task 1 ~~~")
    base_testers = [source_tester]
    base_task = BaseTask(train_source_loader, model, optim_kwargs,
                         base_testers, args, task_id=0)
    base_task.train()

    """Warm-up step for task 2."""
    print("~~~ Training Task 2 ~~~")
    print("~~~ Warm up ~~~")
    testers = [source_tester, target_tester]
    warmup_phase = WarmUpTask(train_target_loader, model, optim_kwargs,
                              testers, args, task_id=1)
    warmup_phase.train()

    """Fine-tune step for task 2."""
    print("~~~ Fine Tune ~~~")
    testers = [source_tester, target_tester]
    orig_model = warmup_phase.orig_model
    fine_tune_phase = FineTuneTask(orig_model, args.lambd, train_target_loader,
                                   model, optim_kwargs, testers, args, task_id=1)
    fine_tune_phase.train()


class TrainingPeriod(ABC):
    """Abstract class representing a period of training during an experiment."""

    def __init__(self, train_loader, model, optim_kwargs, testers, args, task_id):
        self.epochs = args.epochs
        self.device = args.device

        self.train_loader = train_loader
        self.task_id = task_id
        self.testers = testers

        self._init_model(model)
        self._init_optimizer(optim_kwargs)

        self.epoch_train_loss = []
        self.epoch_train_accuracy = []

    def _init_model(self, model):
        self.model = model

    def _init_optimizer(self, optim_kwargs):
        """Initialize the optimizer."""
        self.optimizer = optim.SGD(self.model.parameters(), **optim_kwargs)

    def train(self):
        """Train a model and run all associated tests on it."""
        for epoch in range(1, self.epochs + 1):
            self.train_epoch()
            for t in self.testers:
                t.test(self.model)

    def train_epoch(self):
        # Stats to track
        total_loss = 0
        batches = 0
        accuracy = 0
        n_examples = 0
        correct = 0

        self.model.train()

        # Wrap our data in a tqdm progress bar and iterate through all batches
        pbar = tqdm(self.train_loader, file=sys.stdout)
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            batches += 1

            # Predict on the data
            self.optimizer.zero_grad()
            outputs = self.model(data)

            loss = self.get_loss(data, outputs, target)

            # Update the progress bar with this epoch's loss
            pbar.set_description(f"Loss: {loss.item():0.4f}")
            pbar.update(1)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Check if the max-log prediction was correct
            primary_task_classes = outputs[self.task_id]
            pred = primary_task_classes.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            n_examples += target.shape[0]

        self.epoch_train_loss.append(total_loss / batches)
        self.epoch_train_accuracy.append(correct / (n_examples * 1.0))

        print(f"Epoch Train Loss: {self.epoch_train_loss[-1]:0.4f}")
        print(f"Epoch Train Acc: {self.epoch_train_accuracy[-1]:0.4f}")

    @abstractmethod
    def get_loss(self, data, outputs, target):
        ...


class BaseTask(TrainingPeriod):
    """A class representing a regular training period for one task."""

    def get_loss(self, data, outputs, target):
        """Get the loss from a single epoch."""
        loss = F.nll_loss(outputs[self.task_id], target)
        return loss


class WarmUpTask(TrainingPeriod):
    """A class representing the warm-up phase of training on a task."""

    def __init__(self, *args, **kwargs):
        super(WarmUpTask, self).__init__(*args, **kwargs)

        # We need to freeze the params after initializing the optimizer
        self.model.freeze_params(self.task_id)

    def _init_model(self, model):
        """Initialize the model."""
        self.orig_model = copy.deepcopy(model)
        model.add_prediction_layer(self.device, 10)
        self.model = model

    def get_loss(self, data, outputs, target):
        """Get the loss from a single epoch."""
        loss = F.nll_loss(outputs[self.task_id], target)
        return loss


class FineTuneTask(TrainingPeriod):
    """A class representing the fine-tune phase of training on a task."""

    def __init__(self, orig_model, lambd, *args, **kwargs):
        super(FineTuneTask, self).__init__(*args, **kwargs)

        self.orig_model = orig_model
        self.lambd = lambd

    def _init_model(self, model):
        """Initialize the model."""
        model.unfreeze_params(self.task_id)
        self.model = model

    def _init_optimizer(self, optim_kwargs):
        """Initialize the optimizer."""
        optim_kwargs['lr'] = optim_kwargs['lr'] / 10
        self.optimizer = optim.SGD(self.model.parameters(), **optim_kwargs)

    def get_loss(self, data, outputs, target):
        """Get the loss from a single epoch."""
        old_ouptus = self.orig_model(data)
        old_tasks_loss = 0
        for i, old_out in enumerate(old_ouptus):
            if i != self.task_id:
                old_tasks_loss += F.mse_loss(outputs[i], old_out[i])

        curr_task_loss = F.nll_loss(outputs[self.task_id], target)

        loss = self.lambd * old_tasks_loss + curr_task_loss
        return loss


class Tester:

    def __init__(self, test_loader, args, task_id):
        self.test_loader = test_loader
        self.task_id = task_id
        self.device = args.device

    def test(self, model):
        """Test a model."""
        model.eval()
        test_loss = 0
        correct = 0
        n_examples = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output_class = model(data)[self.task_id]
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
