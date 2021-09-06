import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels, ngf, activation=None):
        super(Discriminator, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=ngf, kernel_size=3, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=ngf)

        self.conv_2 = nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=3, stride=1, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=ngf*2)

        self.conv_3 = nn.Conv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=3, stride=1, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(num_features=ngf*4)

        self.activation = activation
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, labels):

        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.repeat(1, 1, input.size(2), input.size(3))

        input = torch.cat((input, labels), dim=1)

        # 32 x 32 x 11 -> 16 x 16 x 64
        out_conv_1 = self.conv_1(input)
        out_conv_1 = self.batch_norm_1(out_conv_1)
        out_conv_1 = self.relu(out_conv_1)
        out_conv_1 = self.dropout(out_conv_1)
        out_conv_1 = self.maxpool(out_conv_1)

        # 16 x 16 x 64 -> 8 x 8 x 128
        out_conv_2 = self.conv_2(out_conv_1)
        out_conv_2 = self.batch_norm_2(out_conv_2)
        out_conv_2 = self.relu(out_conv_2)
        out_conv_2 = self.dropout(out_conv_2)
        out_conv_2 = self.maxpool(out_conv_2)

        # 8 x 8 x 128 -> 4 x 4 x 256
        out_conv_3 = self.conv_3(out_conv_2)
        out_conv_3 = self.batch_norm_3(out_conv_3)
        out_conv_3 = self.relu(out_conv_3)
        out_conv_3 = self.dropout(out_conv_3)
        out = self.maxpool(out_conv_3)

        if self.activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.activation == 'tanh':
            out = self.tanh(out)

        return out

