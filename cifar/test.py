import torch.nn as nn
import dataset
import torch
import models_cifar
checkpoint = torch.load('/home/lab/BNN/RBNN/cifar/[DIR]/model_best.pth.tar')
model_zoo = 'models_cifar.'
model = nn.DataParallel(eval(model_zoo+'resnet20_bireal_1w1a')(num_classes=10))

model.load_state_dict(checkpoint['state_dict'])
