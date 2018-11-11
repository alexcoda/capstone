import torch.nn.functional as F
import torch.nn as nn
import torch


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

    def __init__(self, lambd, init_task_n_classes=10):
        super(LWFNet, self).__init__()
        self.lambd = lambd

        self.feature_extractor = FeatureExtractor()
        self._label_predictor1 = LabelPredictor(init_task_n_classes)
        self.label_predictors = [self._label_predictor1]

    def forward(self, x):
        # Do the forward pass. Here we predict for each separate label predictor
        x_feature = self.feature_extractor(x)
        x_classes = [lp(x_feature) for lp in self.label_predictors]

        return x_classes

    def add_prediction_layer(self, device, n_classes):
        """Add another layer to the output for predicting on another task."""
        lp = LabelPredictor(n_classes).to(device)
        self.label_predictors.append(lp)
