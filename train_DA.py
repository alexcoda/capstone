import torch.nn.functional as F
import pandas as pd
import sys

from utils import combine_dataframes
from train import TrainingPeriod, Tester
from model import DANet
from tqdm import tqdm


def train_DA(train_loader, test_source_loader, test_target_loader, args):

    # Initialize the model
    model = DANet(args.lambd).to(args.device)
    optim_kwargs = {'lr': args.lr,
                    'momentum': args.momentum}

    # Set-up the tests to run on our model
    # Task-id is always 0 because we share an output layer
    task0_tester = Tester(test_source_loader, args, task_id=0)
    task1_tester = Tester(test_target_loader, args, task_id=0)
    cols = ['train_task_-1', 'test_task_0', 'test_task_1', 'phase']
    results_df = pd.DataFrame(columns=cols)

    """Initial training on both tasks."""
    print("~~~ Training DA on both tasks ~~~")
    testers = [task0_tester, task1_tester]
    # pdb.set_trace()
    da_task = DATrainer(train_loader, model, optim_kwargs, testers, args,
                        task_id=-1, phase_id=-1, df=results_df.copy())
    results_df = da_task.train()

    return combine_dataframes([results_df])


class DATrainer(TrainingPeriod):

    def train_epoch(self):
        """One training epoch for domain adaptation.

        Needs to be a separate function than the base TrainingPeriod because
        our loop also goes over the 'domain' value for each example.
        """
        # Stats to track
        total_loss = 0
        batches = 0
        accuracy = 0
        n_examples = 0
        correct = 0

        self.model.train()

        # Wrap our data in a tqdm progress bar and iterate through all batches
        pbar = tqdm(self.train_loader, file=sys.stdout)
        for (data, target, domain) in pbar:
            data, target, domain = data.to(self.device), target.to(self.device), domain.to(self.device)
            batches += 1

            # Predict on the data
            self.optimizer.zero_grad()
            outputs = self.model(data)
            output_class = outputs[0]

            # Get the total loss
            loss = self.get_loss(data, outputs, target, domain)

            # Update the progress bar with this epoch's loss
            pbar.set_description(f"Loss: {loss.item():0.4f}")
            pbar.update(1)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Check if the max-log prediction was correct
            pred = output_class.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            n_examples += target.shape[0]

        self.epoch_train_loss.append(total_loss / batches)
        self.epoch_train_accuracy.append(correct / (n_examples * 1.0))

        print(f"Epoch Train Loss: {self.epoch_train_loss[-1]:0.4f}")
        print(f"Epoch Train Acc: {self.epoch_train_accuracy[-1]:0.4f}")

    def get_loss(self, data, outputs, target, domain):
        """Get the loss from one batch."""
        output_class, output_domain = outputs

        loss_label = F.nll_loss(output_class, target)
        loss_domain = F.binary_cross_entropy_with_logits(output_domain,
                                                         domain.view(-1, 1))
        return loss_label + loss_domain
