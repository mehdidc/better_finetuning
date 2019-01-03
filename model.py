import torch.nn as nn
from itertools import chain
import torch


class SameModel(nn.Module):
    def __init__(self, base_model, nb_classes=10):
        super().__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Linear(512, nb_classes)

    def parameters(self):
        return self.fc.parameters()

    def forward(self, x):
        return self.base_model(x)


class WiderModel(nn.Module):
    def __init__(self, base_model, nb_classes=10):
        super().__init__()
        self.base_model = base_model
        self.l1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l1_norm_x = L2Norm(64)
        self.l1_norm_h = L2Norm(64)
        self.l2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l2_norm_x = L2Norm(64)
        self.l2_norm_h = L2Norm(64)

        self.l3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l3_norm_x = L2Norm(128)
        self.l3_norm_h = L2Norm(64)
        self.l4 = nn.Sequential(
            nn.Conv2d(256 + 64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l4_norm_x = L2Norm(256)
        self.l4_norm_h = L2Norm(64)

        self.l5_norm_x = L2Norm(512)
        self.l5_norm_h = L2Norm(64)
        self.fc = nn.Linear(512 + 64, nb_classes)

    def parameters(self):
        return chain(
            self.l1.parameters(),
            self.l2.parameters(),
            self.l3.parameters(),
            self.l4.parameters(),
            self.fc.parameters(),
        )

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        h1 = self.l1(x)
        # h1
        x1 = self.base_model.layer1(x)
        # x1

        xn = self.l2_norm_x(x1)
        hn = self.l2_norm_h(h1)
        h = torch.cat((xn, hn), 1)
        h2 = self.l2(h)
        # h2
        x2 = self.base_model.layer2(x1)
        # x2
        xn = self.l3_norm_x(x2)
        hn = self.l3_norm_h(h2)
        h = torch.cat((xn, hn), 1)
        h3 = self.l3(h)
        # h3
        x3 = self.base_model.layer3(x2)
        # x3

        xn = self.l4_norm_x(x3)
        hn = self.l4_norm_h(h3)
        h = torch.cat((xn, hn), 1)
        h4 = self.l4(h)
        # h4
        x4 = self.base_model.layer4(x3)
        # x4

        xn = self.l5_norm_x(x4)
        hn = self.l5_norm_h(h4)
        x5 = torch.cat((xn, hn), 1)
        x = self.base_model.avgpool(x5)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=10):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


if __name__ == "__main__":
    from torchvision.models import resnet18
    import torch

    net = WiderModel(resnet18(pretrained=True))
    x = torch.rand(1, 3, 224, 224)
    y = net(x)
    print(y.size())
