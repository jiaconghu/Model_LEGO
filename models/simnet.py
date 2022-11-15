from torch import nn
from collections import OrderedDict


class SimNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('c1', nn.Conv2d(in_channels, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu1', nn.ReLU()),
                ('s1', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('c2', nn.Conv2d(9, 27, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu2', nn.ReLU()),
                ('s2', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('c3', nn.Conv2d(27, 81, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu3', nn.ReLU())
            ])
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                # ('f4', nn.Linear(254016, num_classes))
                ('f4', nn.Linear(5184, 2048)),
                ('f5', nn.Linear(2048, num_classes))
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def simnet(in_channels, num_classes):
    return SimNet(in_channels, num_classes)


if __name__ == '__main__':
    from torchsummary import summary

    model = simnet(3, 5)
    summary(model, (3, 32, 32))
    print(model)
