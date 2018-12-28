import torch.nn as nn
from itertools import chain
import torch

class Model(nn.Module):
    
    def __init__(self, base_model, nb_classes=10):
        super().__init__()
        self.base_model = base_model
        self.l1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(128+32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(256+64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(512+64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(512 + 64, nb_classes)

    def parameters(self):
        return chain(self.l1.parameters(), self.l2.parameters(), self.l3.parameters(), self.l4.parameters(), self.fc.parameters())

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        h = self.l1(x)
        
        x = self.base_model.layer2(x)
        h = torch.cat((x, h), 1)
        h = self.l2(h)

        x = self.base_model.layer3(x)
        h = torch.cat((x, h), 1)
        h = self.l3(h)
        
        x = self.base_model.layer4(x)
        h = torch.cat((x, h), 1)
        h = self.l4(h)

        x = torch.cat((x, h), 1)
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    from torchvision.models import resnet18
    import torch
    net = Model(resnet18(pretrained=True))
    x = torch.rand(1, 3, 224, 224)
    net(x)
