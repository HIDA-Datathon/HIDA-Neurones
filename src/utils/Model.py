import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import os, sys
sys.path.append(os.path.abspath(".."))
from src.utils.DataLoader import NeutronDataLoader

from torchvision.models.resnet import ResNet, BasicBlock


class GeneratorResNet(ResNet):

    def __init__(self, image_shape, *args, **kwargs):
        super(GeneratorResNet, self).__init__(block=BasicBlock, layers=[3, 0, 0, 0], *args, **kwargs)
        self.image_shape = image_shape

        self.inplanes = 64

        # layers to extend ResNet:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

        del self.fc, self.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.last_conv(x)
        x = self.sigmoid(x)
        return x


class MyModel(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = GeneratorResNet((800, 600))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss_func = nn.MSELoss()
        x, y = batch

        preds = self(x)
        loss = loss_func(preds[:, 0, :, :], y)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        return opt
