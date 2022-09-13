import torch
from torch import nn


def res_net_50(img_channels=3, num_classes=1000):
    return ResNet([3, 4, 6, 3], img_channels, num_classes)


def res_net_101(img_channels=3, num_classes=1000):
    return ResNet([3, 4, 23, 3], img_channels, num_classes)


def res_net_152(img_channels=3, num_classes=1000):
    return ResNet([3, 4, 36, 3], img_channels, num_classes)


class ResNet(nn.Module):
    def __init__(self, layers, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(layers[0], 64, 1)
        self.layer2 = self._make_layer(layers[1], 128, 2)
        self.layer3 = self._make_layer(layers[2], 256, 2)
        self.layer4 = self._make_layer(layers[3], 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, 1, stride),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers.append(
            Block(self.in_channels, out_channels, identity_downsample, stride)
        )
        self.in_channels = out_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity

        x = self.relu(x)

        return x


if __name__ == "__main__":
    net = res_net_152().to("cuda")
    x = torch.randn(2, 3, 224, 224).to("cuda")
    print(net(x).shape)
