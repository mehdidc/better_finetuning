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
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l1_norm_x = L2Norm(64)
        self.l1_norm_h = L2Norm(64)
        self.l2 = nn.Sequential(
            nn.Conv2d(128+64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l2_norm_x = L2Norm(128)
        self.l2_norm_h = L2Norm(64)
        
        self.l3 = nn.Sequential(
            nn.Conv2d(256+64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l3_norm_x = L2Norm(256)
        self.l3_norm_h = L2Norm(64)
        self.l4 = nn.Sequential(
            nn.Conv2d(512+64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l4_norm_x = L2Norm(512)
        self.l4_norm_h = L2Norm(64)
        
        self.l5_norm_x = L2Norm(512)
        self.l5_norm_h = L2Norm(64)
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
        xn = self.l2_norm_x(x)
        hn = self.l2_norm_h(h)
        h = torch.cat((xn, hn), 1)
        h = self.l2(h)

        x = self.base_model.layer3(x)
        xn = self.l3_norm_x(x)
        hn = self.l3_norm_h(h)
        h = torch.cat((xn, hn), 1)
        h = self.l3(h)
        
        x = self.base_model.layer4(x)
        xn = self.l4_norm_x(x)
        hn = self.l4_norm_h(h)
        h = torch.cat((xn, hn), 1)
        h = self.l4(h)
        
        xn = self.l5_norm_x(x)
        hn = self.l5_norm_h(h)
        x = torch.cat((xn, hn), 1)
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale=10):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


if __name__ == '__main__':
    from torchvision.models import resnet18
    import torch
    net = WiderModel(resnet18(pretrained=True))
    x = torch.rand(1, 3, 224, 224)
    y = net(x)
    print(y.size())
