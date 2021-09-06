import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf, activation=None):
        super(Generator, self).__init__()
        # down 1
        self.conv_down_1 = nn.Conv2d(in_channels=in_channels, out_channels=ngf, kernel_size=3, stride=1, padding=1)
        self.conv_down_1_ = nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=1)
        self.conv_down_1_s = nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=4, stride=2, padding=1)
        self.bn_down_1 = nn.BatchNorm2d(num_features=ngf)

        # down 2
        self.conv_down_2 = nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=3, stride=1, padding=1)
        self.conv_down_2_ = nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=3, stride=1, padding=1)
        self.conv_down_2_s = nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=4, stride=2, padding=1)
        self.bn_down_2 = nn.BatchNorm2d(num_features=ngf*2)

        # down 3
        self.conv_down_3 = nn.Conv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=3, stride=1, padding=1)
        self.conv_down_3_ = nn.Conv2d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=3, stride=1, padding=1)
        self.conv_down_3_s = nn.Conv2d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=4, stride=2, padding=1)
        self.bn_down_3 = nn.BatchNorm2d(num_features=ngf*4)

        # bottleneck 1
        self.bottleneck_1 = nn.Conv2d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=3, stride=1, padding=1)
        self.bn_bottleneck_1 = nn.BatchNorm2d(num_features=ngf * 8)

        # bottleneck 2
        self.bottleneck_2 = nn.Conv2d(in_channels=ngf*8, out_channels=ngf*8, kernel_size=3, stride=1, padding=1)
        self.bn_bottleneck_2 = nn.BatchNorm2d(num_features=ngf * 8)

        # bottleneck 3
        self.bottleneck_3 = nn.Conv2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=3, stride=1, padding=1)
        self.bn_bottleneck_3 = nn.BatchNorm2d(num_features=ngf*4)

        # up 1
        self.conv_up_3 = nn.Conv2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=3, stride=1, padding=1)
        self.bn_up_3 = nn.BatchNorm2d(num_features=ngf*4)
        self.conv_up_3_ = nn.Conv2d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=3, stride=1, padding=1)
        self.conv_up_3_s = nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=4, stride=2, padding=1)

        # up 2
        self.conv_up_2 = nn.Conv2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=3, stride=1, padding=1)
        self.bn_up_2 = nn.BatchNorm2d(num_features=ngf*2)
        self.conv_up_2_ = nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=3, stride=1, padding=1)
        self.conv_up_2_s = nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=4, stride=2, padding=1)

        # up 2
        self.conv_up_1 = nn.Conv2d(in_channels=ngf*2, out_channels=ngf, kernel_size=3, stride=1, padding=1)
        self.bn_up_1 = nn.BatchNorm2d(num_features=ngf)
        self.conv_up_1_ = nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=1)
        self.conv_up_1_s = nn.ConvTranspose2d(in_channels=ngf, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

        self.activation = activation
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_unpool = nn.MaxUnpool2d(2, stride=2)
        self.dropout = nn.Dropout2d(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, labels):

        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.repeat(1, 1, input.size(2), input.size(3))


        input = torch.cat((input, labels), dim=1)

        # -----------------------------------------------------------------------------------
        #                                       Downsample
        # -----------------------------------------------------------------------------------

        # ----------------- down block 1 ------------------
        # 32 x 32 x in_channels -> 16 x 16 x ngf
        down_1 = self.conv_down_1(input)
        down_1 = self.bn_down_1(down_1)
        down_1 = self.relu(down_1)

        down_1 = self.conv_down_1_(down_1)
        down_1 = self.bn_down_1(down_1)
        down_1 = self.relu(down_1)

        down_1 = self.dropout(down_1)
        down_1 = self.conv_down_1_s(down_1)
        down_1 = self.relu(down_1)

        # ----------------- down block 2 ------------------
        # 16 x 16 x ngf -> 8 x 8 x ngf*2
        down_2 = self.conv_down_2(down_1)
        down_2 = self.bn_down_2(down_2)
        down_2 = self.relu(down_2)

        down_2 = self.conv_down_2_(down_2)
        down_2 = self.bn_down_2(down_2)
        down_2 = self.relu(down_2)

        down_2 = self.dropout(down_2)
        down_2 = self.conv_down_2_s(down_2)
        down_2 = self.relu(down_2)

        # ----------------- down block 3 ------------------
        # 8 x 8 x ngf*2 -> 4 x 4 x ngf*4
        down_3 = self.conv_down_3(down_2)
        down_3 = self.bn_down_3(down_3)
        down_3 = self.relu(down_3)

        down_3 = self.conv_down_3_(down_3)
        down_3 = self.bn_down_3(down_3)
        down_3 = self.relu(down_3)

        down_3 = self.dropout(down_3)
        down_3 = self.conv_down_3_s(down_3)
        down_3 = self.relu(down_3)

        # -----------------------------------------------------------------------------------
        #                                     Bottleneck
        # -----------------------------------------------------------------------------------
        # ----------------- bottleneck 1 ------------------
        # 4 x 4 x ngf*4 -> 4 x 4 x ngf*8
        bottleneck_1 = self.bottleneck_1(down_3)
        bottleneck_1 = self.bn_bottleneck_1(bottleneck_1)
        bottleneck_1 = self.relu(bottleneck_1)
        bottleneck_1 = self.dropout(bottleneck_1)

        # ----------------- bottleneck 2 ------------------
        # 4 x 4 x ngf*8 -> 4 x 4 x ngf*8
        bottleneck_2 = self.bottleneck_2(bottleneck_1)
        bottleneck_2 = self.bn_bottleneck_2(bottleneck_2)
        bottleneck_2 = self.relu(bottleneck_2)
        bottleneck_2 = self.dropout(bottleneck_2)

        # ----------------- bottleneck 3 ------------------
        # 4 x 4 x ngf*8 -> 4 x 4 x ngf*4
        bottleneck_3 = self.bottleneck_3(bottleneck_2)
        bottleneck_3 = self.bn_bottleneck_3(bottleneck_3)
        bottleneck_3 = self.relu(bottleneck_3)
        bottleneck_3 = self.dropout(bottleneck_3)

        # -----------------------------------------------------------------------------------
        #                                       Upsample
        # -----------------------------------------------------------------------------------

        # ----------------- up block 3 ------------------
        # 4 x 4 x ngf*4 -> 4 x 4 x ngf*8
        up_3 = torch.cat((bottleneck_3, down_3), dim=1)

        # 4 x 4 x ngf*8 -> # 4 x 4 x ngf*2
        up_3 = self.conv_up_3(up_3)
        up_3 = self.bn_up_3(up_3)
        up_3 = self.leaky_relu(up_3)

        up_3 = self.conv_up_3_(up_3)
        up_3 = self.bn_up_3(up_3)
        up_3 = self.leaky_relu(up_3)

        up_3 = self.dropout(up_3)
        up_3 = self.conv_up_3_s(up_3)
        up_2 = self.leaky_relu(up_3)

        # ----------------- up block 2 ------------------
        # 4 x 4 x ngf*2 -> 4 x 4 x ngf*4
        up_2 = torch.cat((up_2, down_2), dim=1)

        # 4 x 4 x ngf*4 -> 4 x 4 x ngf
        up_2 = self.conv_up_2(up_2)
        up_2 = self.bn_up_2(up_2)
        up_2 = self.leaky_relu(up_2)

        up_2 = self.conv_up_2_(up_2)
        up_2 = self.bn_up_2(up_2)
        up_2 = self.leaky_relu(up_2)

        up_2 = self.dropout(up_2)
        up_2 = self.conv_up_2_s(up_2)
        up_1 = self.leaky_relu(up_2)

        # ----------------- up block 2 ------------------
        # 4 x 4 x ngf*2 -> 4 x 4 x ngf*4
        up_1 = torch.cat((up_1, down_1), dim=1)

        # 4 x 4 x ngf*4 -> 4 x 4 x ngf
        up_1 = self.conv_up_1(up_1)
        up_1 = self.bn_up_1(up_1)
        up_1 = self.leaky_relu(up_1)

        up_1 = self.conv_up_1_(up_1)
        up_1 = self.bn_up_1(up_1)
        up_1 = self.leaky_relu(up_1)

        up_1 = self.dropout(up_1)
        up_1 = self.conv_up_1_s(up_1)
        out = self.leaky_relu(up_1)

        if self.activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.activation == 'tanh':
            out = self.tanh(out)

        return out

