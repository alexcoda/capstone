"""Functions/classes used across all experiments."""
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import sys

from tqdm import tqdm
from abc import ABC, abstractmethod


class TrainingPeriod(ABC):
    """Abstract class representing a period of training during an experiment."""

    def __init__(self, train_loader, model, optim_kwargs, testers, args,
                 task_id, phase_id, df, vocab_size=None):
        self.epochs = args.epochs
        self.device = args.device
        self.vocab_size = vocab_size

        self.text_mode = args.text_mode

        self.train_loader = train_loader
        self.task_id = task_id
        self.phase_id = phase_id
        self.testers = testers

        self._init_model(model)
        self._init_optimizer(optim_kwargs)

        self.epoch_train_loss = []
        self.epoch_train_accuracy = []
        self.epoch_test_accuracy = []
        self.df = df

    def _init_model(self, model):
        self.model = model

    def _init_optimizer(self, optim_kwargs):
        """Initialize the optimizer."""
        self.optimizer = optim.SGD(self.model.parameters(), **optim_kwargs)

    def train(self):
        """Train a model and run all associated tests on it."""
        for epoch in range(1, self.epochs + 1):
            self.train_epoch()
            test_acc = [t.test(self.model) for t in self.testers]
            self.epoch_test_accuracy.append(test_acc)

        self._compile_df()
        return self.df

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
        for batch in pbar:
            batch = [item.to(self.device) for item in batch]

        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            batches += 1

            # Predict on the data
            self.optimizer.zero_grad()
            outputs = self.model(data)

            loss = self.get_loss(data, outputs, target, text_mode=self.text_mode)

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
    def get_loss(self, data, outputs, target, text_mode=0):
        ...

    def _compile_df(self):
        """Compile the run results into a final DataFrame."""
        task_ids = [i for i in range(len(self.testers))]
        test_vals = np.array(self.epoch_test_accuracy).T

        for i in task_ids:
            self.df[f"test_task_{1 - i}"] = test_vals[i]
        self.df[f"train_task_{1 - self.task_id}"] = self.epoch_train_accuracy
        self.df['phase'] = self.phase_id


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
        test_acc = correct / n_examples
        perc_correct = 100. * test_acc

        print(f"\nTest set: Average loss: {test_loss:0.4f},"
              f"Accuracy: {correct}/{n_examples} ({perc_correct:.0f}%)\n")
        return test_acc
