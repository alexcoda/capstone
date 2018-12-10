"""Functions/classes used for LwF experiments."""
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import copy

# Local imports
from model import LWFNet
from utils import combine_dataframes
from train import TrainingPeriod, Tester


def train_LWF(train_source_loader, train_target_loader,
              test_target_loader, test_source_loader, args, vocab_size=None):

    # Initialize the model
    model = LWFNet(text_mode=args.text_mode, vocab_size=vocab_size).to(args.device)
    optim_kwargs = {'lr': args.lr,
                    'momentum': args.momentum,
                    'weight_decay': 0.0005}

    # Set-up the tests to run on our model
    source_tester = Tester(test_source_loader, args, task_id=0)
    target_tester = Tester(test_target_loader, args, task_id=1)
    cols = ['train_task_0', 'train_task_1', 'test_task_0', 'test_task_1',
            'phase']
    results_df = pd.DataFrame(columns=cols)

    """Initial training on task 1."""
    print("~~~ Training Task 1 ~~~")
    base_testers = [source_tester]
    base_task = BaseTask(train_source_loader, model, optim_kwargs, base_testers,
                         args, task_id=0, phase_id=0, df=results_df.copy(), vocab_size=vocab_size)
    base_df = base_task.train()

    """Warm-up step for task 2."""
    print("~~~ Training Task 2 ~~~")
    print("~~~ Warm up ~~~")
    testers = [source_tester, target_tester]
    warmup_phase = WarmUpTask(train_target_loader, model, optim_kwargs, testers,
                              args, task_id=1, phase_id=1, df=results_df.copy(), vocab_size=vocab_size)
    warmup_df = warmup_phase.train()

    """Fine-tune step for task 2."""
    print("~~~ Fine Tune ~~~")
    testers = [source_tester, target_tester]
    orig_model = warmup_phase.orig_model
    fine_tune_phase = FineTuneTask(orig_model, args.lambd, train_target_loader,
                                   model, optim_kwargs, testers, args,
                                   task_id=1, phase_id=2, df=results_df.copy(), vocab_size=vocab_size)
    fine_tune_df = fine_tune_phase.train()

    return combine_dataframes([base_df, warmup_df, fine_tune_df])


class BaseTask(TrainingPeriod):
    """A class representing a regular training period for one task."""

    def get_loss(self, data, outputs, target, text_mode=0):
        """Get the loss from a single epoch."""
        if text_mode:
            loss = F.cross_entropy(outputs[self.task_id], target)
        else:
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

    def get_loss(self, data, outputs, target, text_mode=0):
        """Get the loss from a single epoch."""
        if text_mode:
            loss = F.cross_entropy(outputs[self.task_id], target)
        else:
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

    def get_loss(self, data, outputs, target, text_mode=0):
        """Get the loss from a single epoch."""
        old_ouptus = self.orig_model(data)
        old_tasks_loss = 0
        for i, old_out in enumerate(old_ouptus):
            if i != self.task_id:
                old_tasks_loss += F.mse_loss(outputs[i], old_out[i])

        if text_mode:
            curr_task_loss = F.cross_entropy(outputs[self.task_id], target)
        else:
            curr_task_loss = F.nll_loss(outputs[self.task_id], target)

        loss = self.lambd * old_tasks_loss + curr_task_loss
        return loss
