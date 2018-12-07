import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import utils

from model import EWCNet

torch.manual_seed(0)

def train_ewc(train_datasets, test_datasets, epochs_per_task=2,
          batch_size=64, consolidate=False,
          fisher_estimation_sample_size=1024,
          lr=1e-3, weight_decay=1e-5, lamda=5000,
          cuda=False):

    print(consolidate)
    model = EWCNet(flatten=False)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    model.train()
    precs = []

    for task, data_loader in enumerate(train_datasets, 1):
        for epoch in range(1, epochs_per_task+1):

            data_stream = tqdm(enumerate(data_loader, 1))
            loader_size = len(data_loader)

            for batch_index, (x, y) in data_stream:

                dataset_size = len(data_loader.dataset)
                dataset_batches = len(data_loader)
                previous_task_iteration = sum([
                    epochs_per_task * len(d) // batch_size for d in
                    train_datasets[:task-1]
                ])
                current_task_iteration = (
                    (epoch-1)*dataset_batches + batch_index
                )
                iteration = (
                    previous_task_iteration +
                    current_task_iteration
                )

                if model.flatten:
                    data_size = len(x)
                    x = x.view(data_size, -1)
                x = Variable(x).cuda() if cuda else Variable(x)
                y = Variable(y).cuda() if cuda else Variable(y)

                optimizer.zero_grad()
                scores = model(x)
                ce_loss = criterion(scores, y)
                ewc_loss = model.get_ewc_loss(lamda, cuda=cuda)
                loss = ce_loss + ewc_loss
                loss.backward()
                optimizer.step()
                _, predicted = scores.max(1)
                precision = (predicted == y).sum().data.item() / len(x)

                data_stream.set_description((
                    'task: {task}/{tasks} | '
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'prec: {prec:.4} | '
                    'loss => '
                    'ce: {ce_loss:.4} / '
                    'ewc: {ewc_loss:.4} / '
                    'total: {loss:.4} / '
                    'all: {precs}'
                ).format(
                    task=task,
                    tasks=len(train_datasets),
                    epoch=epoch,
                    epochs=epochs_per_task,
                    trained=batch_index*batch_size,
                    total=dataset_size,
                    progress=(100.*batch_index/dataset_batches),
                    prec=precision,
                    ce_loss=ce_loss.data.item(),
                    ewc_loss=ewc_loss.data.item(),
                    loss=loss.data.item(),
                    precs=precs,
                ))

                if batch_index == (loader_size-1):
                    names = [
                        'task {}'.format(i+1) for i in
                        range(len(train_datasets))
                    ]
                    precs = [
                        utils.validate(
                            model, test_datasets[i], test_size=len(test_datasets[i]),
                            cuda=cuda, verbose=False,
                        ) if i+1 <= task else 0 for i in
                        range(len(train_datasets))
                    ]
                    title = (
                        'precision (consolidated)' if consolidate else
                        'precision'
                    )

        if consolidate:
            model.consolidate(model.estimate_fisher(
                data_loader, fisher_estimation_sample_size
            ))
