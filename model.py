import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = 1 * 28 * 28  # input spatial dimension of images
hidden_size = 128  # width of hidden layer
output_size = 10  # number of output neurons


class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        # ------------------
        self.conv1 = nn.Conv2d(1, 10, 5, stride=1, padding=0)  # first convolutional layer
        self.conv2 = nn.Conv2d(10, 20, 5, stride=1, padding=0)  # second convolutional layer
        self.fc = nn.Linear(320, 128)  # third fully-connected layer

        self.out = torch.nn.Linear(128, 10)
        self.log_softmax = torch.nn.LogSoftmax()
        # ------------------

    def forward(self, x):
        # Input image is of shape [batch_size, 1, 28, 28]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # Need to flatten to [batch_size, 784] before feeding to fc1
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = self.out(x)
        x = self.log_softmax(x)

        # ------------------

        y_output = x

        return y_output
        # ------------------
