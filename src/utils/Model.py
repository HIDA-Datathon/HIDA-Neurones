import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.abspath(".."))
from src.utils.DataLoader import NeutronDataLoader
from torchvision.models.resnet import ResNet, BasicBlock
from pytorch_lightning.metrics.functional.classification import dice_score
import numpy as np
from argparse import ArgumentParser


class GeneratorResNet(ResNet):

    def __init__(self, image_shape, num_classes=21, *args, **kwargs):
        super(GeneratorResNet, self).__init__(block=BasicBlock, layers=[3, 0, 0, 0], *args, **kwargs)
        self.image_shape = image_shape

        self.inplanes = 64

        # layers to extend ResNet:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.last_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax2d()

        del self.fc, self.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.last_conv(x)
        # x = self.softmax(x)
        return x


class MyModel(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = GeneratorResNet((800, 600))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        stage = "Training"
        x, y = batch

        preds = self(x)
        y_2d = y.argmax(axis=1).long()
        loss = F.cross_entropy(input=preds, target=y_2d) + (1 - dice_score(pred=preds, target=y_2d))
        self.logger.experiment.log_metric(f"Loss {stage}", loss, step=self.global_step)
        self.log("Train Loss", loss, on_step=True)

        # if True:
        #     self._log_step_figures(x, y_2d, preds)

        return loss

    def validation_step(self, batch, batch_idx):
        stage = "Validation"
        x, y = batch
        preds = self(x)
        y_2d = y.argmax(axis=1).long()
        loss = F.cross_entropy(input=preds, target=y_2d) + (1 - dice_score(pred=preds, target=y_2d))
        self.log("Valid Loss", loss, on_step=True)
        self.logger.experiment.log_metric(f"Loss {stage}", loss, step=self.global_step)

        score = dice_score(preds, y_2d)
        self.log("Valid Score", score, on_step=True)
        self.logger.experiment.log_metric(f"Dice score {stage}", score, step=self.global_step)

        if True:
            self._log_step_figures(x, y_2d, preds)

        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        return opt

    def _log_step_figures(self, x, y, y_hat):

        fig, ax = plt.subplots()
        im = ax.imshow(y[0, ].detach().cpu().numpy(), cmap="inferno")
        plt.colorbar(im)
        self.logger.experiment.log_figure(figure=fig, figure_name=f"Target",
                                          step=self.global_step)

        plt.close("all")

        fig2, ax = plt.subplots()
        im2 = ax.imshow(y_hat.argmax(dim=1)[0, ].detach().cpu().numpy(), cmap="inferno")
        plt.colorbar(im2)
        self.logger.experiment.log_figure(figure=fig2, figure_name=f"Prediction",
                                          step=self.global_step)

        plt.close("all")
        fig3, ax = plt.subplots()
        im3 = ax.imshow(np.moveaxis(x[0, ].detach().cpu().numpy(), 0, -1), cmap="inferno")
        self.logger.experiment.log_figure(figure=fig3, figure_name=f"Image",
                                          step=self.global_step)

        plt.close("all")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0002, help="adam: learning rate")
        return parser
