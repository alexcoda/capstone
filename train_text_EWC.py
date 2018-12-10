import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import utils

from text_model import TextEWCNet
import numpy as np

torch.manual_seed(0)

def train_text_ewc(train_datasets, test_datasets,
          vocab_size,
          epochs_per_task=5,
          batch_size=64, consolidate=False,
          fisher_estimation_sample_size=2048,
          lr=1e-3, weight_decay=1e-5, lamda=100000,
          cuda=False):

    
    model = TextEWCNet(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    num_tasks = len(test_datasets)
    model.train()
    precs = []

    for task, data_loader in enumerate(train_datasets, 1):
        for epoch in range(1, epochs_per_task+1):

            data_stream = tqdm(enumerate(data_loader, 1))
            loader_size = len(data_loader)

            epoch_loss = 0
            epoch_acc = 0
            for batch_index, (x, y) in data_stream:

                optimizer.zero_grad()
                scores = model(x)
                ce_loss = criterion(scores, y)
                ewc_loss = model.get_ewc_loss(lamda, cuda=cuda)
                loss = ce_loss + ewc_loss
                loss.backward()
                optimizer.step()

                acc = binary_accuracy(scores, y)
                epoch_loss += loss.item()
                epoch_acc += acc

                data_stream.set_description((
                    'task: {task}/{tasks} | '
                    'epoch: {epoch}/{epochs} | '
                    'progress: ({progress:.0f}%) | '
                    'loss => '
                    'ce: {ce_loss:.4} / '
                    'ewc: {ewc_loss:.4} / '
                ).format(
                    task=task,
                    tasks=len(train_datasets),
                    epoch=epoch,
                    epochs=epochs_per_task,
                    progress=(100.*batch_index/len(data_loader)),
                    ce_loss=ce_loss.data.item(),
                    ewc_loss=ewc_loss.data.item(),
                    loss=loss.data.item(),
                ))

            train_loss, train_acc = epoch_loss / len(data_loader), epoch_acc / len(data_loader)

            test_losses = []
            test_accs = []
            for i in range(0, num_tasks):
                test_loss, test_acc = evaluate(model, test_datasets[i], criterion)
                test_losses.append(test_loss)
                test_accs.append(test_acc)

            test_losses = ", ".join([str(l) for l in test_losses])
            test_accs = "% ".join([str(v*100)[:4] for v in test_accs]) + "%"
            print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {test_losses} | Val. Acc: {test_accs} |')

        #model.reinit_embedding()
        if consolidate:
            model.consolidate(model.estimate_fisher(
                data_loader, fisher_estimation_sample_size
            ))

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    num_predictions = np.argmax(preds.data.numpy(), axis=1).tolist()
    y = y.numpy().tolist()
    correct = [(1 if x==y else 0) for x,y in zip(num_predictions, y)]
    acc = sum(correct)/len(correct)
    return acc

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for x,y in iterator:

            predictions = model(x).squeeze(1)
            #y = y.type(torch.FloatTensor)
            
            loss = criterion(predictions, y)
            
            acc = binary_accuracy(predictions, y)

            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)