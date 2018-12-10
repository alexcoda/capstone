import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import reduce
from torch import nn
from torch import autograd
from torch.autograd import Variable
import utils

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        
        self.emb_dim = embedding_dim
        self.vocab_size = vocab_size
        self.init_embedding()
        self.fc = nn.Linear(embedding_dim, output_dim)

    def init_embedding(self):
    	self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        
    def forward(self, x):
        
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        
        return self.fc(pooled)

class TextEWCNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim=100, output_dim=2):
        super().__init__()

        self.fast_text = FastText(vocab_size, embedding_dim, output_dim)
        self.flatten = False

    def reinit_embedding(self):
    	self.fast_text.init_embedding()

    def forward(self, x):

        pred = self.fast_text(x)
        return pred

    def estimate_fisher(self, data_loader, sample_size, batch_size=32):

        loglikelihoods = []
        for x, y in data_loader:
            batch_size = x.size(1)
            logits = F.log_softmax(self(x), dim=1)

            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append(
                logits[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break

        loglikelihood = torch.cat(loglikelihoods).mean(0)
        loglikelihood_grads = autograd.grad(loglikelihood, self.parameters())
        parameter_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        return {n: g**2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            self.register_buffer('{}_estimated_fisher'
                                 .format(n), fisher[n].data.clone())

    def get_ewc_loss(self, lamda, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                n = n.replace('.', '__')
                mean = getattr(self, '{}_estimated_mean'.format(n))
                fisher = getattr(self, '{}_estimated_fisher'.format(n))
                mean = Variable(mean)
                fisher = Variable(fisher)
                losses.append((fisher * (p-mean)**2).sum())
            return (lamda/2)*sum(losses)
        except AttributeError:
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda