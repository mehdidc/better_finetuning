from fastai.vision import *
from model import Model
from torchvision.models import resnet18
import torch
net = Model(resnet18(pretrained=True))
path = untar_data(URLs.DOGS)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data, net, metrics=accuracy)
