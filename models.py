# Model classes defined here

import torch
import torch.nn.functional as F
import sys

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        # define two feedforward layers
        self.linear1 = torch.nn.Linear(28*28, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        # define 2 convolutional layers
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(n1_chan, 10, kernel_size=n2_kern, stride=2)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 8)
        # reshape output to correct size
        x = x.view(x.shape[0], 10)
        return x

class BestNN(torch.nn.Module):
    def __init__(self, n1_chan, n2_chan, n1_kern, n2_kern, lin1_trans, lin2_trans):
        super(BestNN, self).__init__()
        self.n2_chan = n2_chan

        # conv + maxpool sublayer
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.bn1 = torch.nn.BatchNorm2d(n1_chan)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(2, stride=2)

        # conv + maxpool sublayer
        self.conv2 = torch.nn.Conv2d(n1_chan, n2_chan, kernel_size=n2_kern)
        self.bn2 = torch.nn.BatchNorm2d(n2_chan)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(2, stride=2)

        # fully connected layer
        self.linear1 = torch.nn.Linear(n2_chan*4*4, lin1_trans)
        self.bn3 = torch.nn.BatchNorm1d(lin1_trans)
        self.relu3 = torch.nn.ReLU()

        # fully connected layer
        self.linear2 = torch.nn.Linear(lin1_trans, lin2_trans)
        self.bn4 = torch.nn.BatchNorm1d(lin2_trans)
        self.relu4 = torch.nn.ReLU()

        # final output layer
        self.output = torch.nn.Linear(lin2_trans, 10)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 28, 28)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.reshape(-1, self.n2_chan*4*4)
        x = self.linear1(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.linear2(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.output(x)

        return x
