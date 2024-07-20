#!/usr/bin/env python3
# Purpose: Defines the earthquake/blast neural network's architecture.
# Copyright: Ben Baker (University of Utah) distributed under the MIT license.
import torch
import os
import torchvision.transforms as transforms

class CNN(torch.nn.Module):
    def __init__(self,num_classes=2, depth=1,
                 n_channels : int = 3,  # Three-channel scalograms
                 input_height : int = 51, # Number of frequency bins (originally 40)
                 input_width : int = 72,  # Number of time windows (originally 48)
                 kernel_size : int = 3,
                 random_seed : int = None):
        super(CNN, self).__init__()
        from torch import nn
        self.n_input_channels = n_channels
        self.input_width = input_width
        self.input_height = input_height
        filters = [18, 36, 54, 54]
        self.relu = torch.nn.ReLU()
        p = kernel_size//2
        self.conv2d_1 = nn.Conv2d(in_channels = n_channels,
                                  out_channels = filters[0],
                                  kernel_size = kernel_size,
                                  padding = p)
        self.batch_norm_1 = nn.BatchNorm2d(filters[0], eps = 1e-05, momentum = 0.1)
        # H_out = [40 + 2*k]/2 = 20
        # W_out = [48 + 2*k]/2 = 24
        self.conv2d_2 = nn.Conv2d(in_channels = filters[0],
                                  out_channels = filters[1],
                                  kernel_size = kernel_size,
                                  padding = p)
        self.batch_norm_2 = nn.BatchNorm2d(filters[1], eps = 1e-05, momentum = 0.1)
        # H_out = [20 + 2*k]/2 = 10
        # W_out = [24 + 2*k]/2 = 12
        self.conv2d_3 = nn.Conv2d(in_channels = filters[1],
                                  out_channels = filters[2],
                                  kernel_size = kernel_size,
                                  padding = p)
        self.batch_norm_3 = nn.BatchNorm2d(filters[2], eps = 1e-05, momentum = 0.1)
        # H_out = [10 + 2*k]/2 = 5
        # W_out = [12 + 2*k]/2 = 6
        self.conv2d_4 = nn.Conv2d(in_channels = filters[2],
                                  out_channels = filters[3],
                                  kernel_size = kernel_size,
                                  padding = p)
        self.batch_norm_4 = nn.BatchNorm2d(filters[3], eps=1e-05, momentum=0.1)
        # H_out = [5 + 2*k]/2 = 2
        # W_out = [6 + 2*k]/2 = 3
        _h = (input_height//2)//2//2//2
        _w = (input_width//2)//2//2//2
        fc1_input_length = filters[-1]*_h*_w
        fc1_output_length = fc1_input_length//2 
        self.linear_1 = nn.Linear(fc1_input_length, fc1_output_length)
        self.batch_norm_5 = nn.BatchNorm1d(fc1_output_length, eps=1e-05, momentum=0.1)
        # Add one more layer for some non-linearity
        self.linear_2 =  nn.Linear(fc1_output_length, num_classes)

        # Initialize all weights
        if (random_seed is not None):
            torch.manual_seed(random_seed) 
        nn.init.xavier_normal_(self.conv2d_1.weight)
        nn.init.zeros_(self.conv2d_1.bias)
        nn.init.xavier_normal_(self.conv2d_2.weight)
        nn.init.zeros_(self.conv2d_2.bias)
        nn.init.xavier_normal_(self.conv2d_3.weight)
        nn.init.zeros_(self.conv2d_3.bias)
        nn.init.xavier_normal_(self.conv2d_4.weight)
        nn.init.zeros_(self.conv2d_4.bias)
        nn.init.xavier_normal_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_normal_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

        # For convenience - put it into a layer so the `forward' evaluation method is simple
        self.network = nn.Sequential(self.conv2d_1,
                                     nn.ReLU(),
                                     self.batch_norm_1,
                                     nn.MaxPool2d(kernel_size = 2, stride = 2),
                                     # second layer
                                     self.conv2d_2,
                                     nn.ReLU(),
                                     self.batch_norm_2,
                                     nn.MaxPool2d(kernel_size = 2, stride = 2),
                                     # third layer
                                     self.conv2d_3,
                                     nn.ReLU(),
                                     self.batch_norm_3,
                                     nn.MaxPool2d(kernel_size = 2, stride = 2),
                                     # fourth layer
                                     self.conv2d_4,
                                     nn.ReLU(),
                                     self.batch_norm_4,
                                     nn.MaxPool2d(kernel_size = 2, stride = 2),
                                     # flatten
                                     nn.Flatten(1),
                                     # fully connected layer 1,
                                     self.linear_1,
                                     nn.ReLU(),
                                     self.batch_norm_5,
                                     # fully connected layer 2
                                     self.linear_2)
        # Count size of model
        self.total_number_of_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Given the scalogram features in x this applies the CNN.
        """
        y = self.network.forward(x)
        if (not self.training):
            return torch.sigmoid(y)
        else:
            return y

    def get_input_channels(self) -> int:
        return self.n_input_channels

    def get_input_height(self) -> int:
        return self.input_height

    def get_input_width(self) -> int:
        return self.input_width

    def get_total_number_of_parameters(self) -> int:
        return self.total_number_of_parameters
    def load_from_jit(self, model_path : str):
        if (not os.path.exists(model_path)):
            raise Exception("Model file {} does not exist".format(model_path))
        self.load_state_dict(torch.load(model_path)["state_dict"])


class CNN1:
    base = CNN
    args = list()
    kwargs = {"depth": 1}
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


if __name__ == "__main__":
    import numpy as np
    # N, C, H, W
    x = np.ndarray.astype(np.ones([11, 3, 51, 72]), dtype = np.float32)
    x = torch.from_numpy(x)
    #print(x.shape)
    cnn = CNN(random_seed = 852)
    cnn.train()
    print("Number of parameters:", cnn.get_total_number_of_parameters())
    y = cnn.forward(x)
    print("Output shape:", y.shape)
    print(y)
