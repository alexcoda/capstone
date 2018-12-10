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
        # self.fc = nn.Linear(embedding_dim, output_dim)

    def init_embedding(self):
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        
    def forward(self, x):
        
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        
        # return self.fc(pooled)
        return pooled

class MLP(nn.Module):

    def __init__(self, input_size, output_size, 
                    hidden_size=400,
                    hidden_layer_num=2,
                    hidden_dropout_prob=.5,
                    input_dropout_prob=.2):

        super(MLP, self).__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size

        self.layers = nn.ModuleList([
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout_prob),
            *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout_prob)) * self.hidden_layer_num),
            nn.Linear(self.hidden_size, self.output_size)
        ])

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 48 * 4 * 4)
        return x


class LabelPredictor(nn.Module):

    def __init__(self, n_classes=10):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.op = nn.Linear(100, n_classes)

    def forward(self, x):
        x_class = F.relu(self.fc1(x))
        x_class = F.dropout(x_class, training=self.training)
        x_class = F.relu(self.fc2(x_class))
        x_class = self.op(x_class)
        x_class = F.log_softmax(x_class, dim=1)
        return x_class

class Predictor(nn.Module):

    def __init__(self, embedding_dim, output_dim):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x_class = self.fc(x)
        return x_class

class GradReverse(torch.autograd.Function):

    def __init__(self, lambd):
        super(GradReverse, self)
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class DomainClassifier(nn.Module):

    def __init__(self, lambd):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(768, 100)
        self.fc2 = nn.Linear(100, 1)
        self.drop = nn.Dropout2d(0.25)
        self.lambd = lambd

    def forward(self, x):
        x = grad_reverse(x, self.lambd)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)


class DANet(nn.Module):

    def __init__(self, lambd):
        super(DANet, self).__init__()
        self.lambd = lambd

        self.feature_extractor = FeatureExtractor()
        self.label_predictor = LabelPredictor()
        self.domain_classifier = DomainClassifier(self.lambd)

    def forward(self, x):
        x_feature = self.feature_extractor(x)

        x_class = self.label_predictor(x_feature)

        x_domain = self.domain_classifier(x_feature)

        return x_class, x_domain


class LWFNet(nn.Module):

    def __init__(self, init_task_n_classes=10, embedding_dim=100, output_dim=2, vocab_size=None, text_mode=False):
        super(LWFNet, self).__init__()

        self.text_mode = text_mode
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        # print(self.text_mode, self.embedding_dim, self.output_dim, self.vocab_size)
        # s = input()

        if text_mode:
            self.feature_extractor = FastText(vocab_size, embedding_dim, output_dim)
            self.label_predictors = nn.ModuleList([Predictor(embedding_dim, output_dim)])
        else:
            self.feature_extractor = FeatureExtractor()
            self.label_predictors = nn.ModuleList([LabelPredictor(init_task_n_classes)])


    def forward(self, x):
        # Do the forward pass. Here we predict for each separate label predictor
        x_feature = self.feature_extractor(x)
        x_classes = [lp(x_feature) for lp in self.label_predictors]

        return x_classes

    def add_prediction_layer(self, device, n_classes):
        """Add another layer to the output for predicting on another task."""
        if self.text_mode:
            self.label_predictors.append(Predictor(self.embedding_dim, self.output_dim).to(device))
        else:
            self.label_predictors.append(LabelPredictor(n_classes).to(device))


    def freeze_params(self, task_id):
        """Freeze all parameters of this model except for the current task."""
        self._toggle_params(task_id, freeze=True)

    def unfreeze_params(self, task_id):
        """Unfreeze all parameters of this model except for the current task."""
        self._toggle_params(task_id, freeze=False)

    def _toggle_params(self, task_id, freeze):
        """Toggle whether params are frozen."""
        requires_grad = not freeze
        for param in self.feature_extractor.parameters():
            param.requires_grad = requires_grad
        for i, layer in enumerate(self.label_predictors):
            if i != task_id:
                for param in layer.parameters():
                    param.requires_grad = requires_grad

class EWCNet(nn.Module):
    def __init__(self, input_size=784, output_size=10, flatten=False):
        super().__init__()

        self.flatten = flatten
        if self.flatten:
            self.MLP = MLP(input_size, output_size)
        else:
            self.feature_extractor = FeatureExtractor()
            self.predictor = Predictor()

    @property
    def name(self):
        return (
            'MLP'
            '-in{input_size}-out{output_size}'
            '-h{hidden_size}x{hidden_layer_num}'
            '-dropout_in{input_dropout_prob}_hidden{hidden_dropout_prob}'
        ).format(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            hidden_layer_num=self.hidden_layer_num,
            input_dropout_prob=self.input_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

    def forward(self, x):

        if self.flatten:
            pred = self.MLP(x)
        else:
            features = self.feature_extractor(x)
            pred = self.predictor(features)

        return pred

    def estimate_fisher(self, data_loader, sample_size, batch_size=32):

        loglikelihoods = []
        for x, y in data_loader:
            batch_size = len(x)

            if self.flatten:
                x = x.view(batch_size, -1)

            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append(
                F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
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
