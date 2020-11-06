import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.abspath(".."))
from pytorch_lightning.metrics.functional.classification import dice_score
import numpy as np
from argparse import ArgumentParser
from torchvision.models.segmentation import fcn_resnet50


class MyModel(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = fcn_resnet50(pretrained=True, num_classes=21)

    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        stage = "Training"
        x, y = batch

        preds = self(x)
        y_2d = y.argmax(axis=1).long()
        loss = F.cross_entropy(input=preds, target=y_2d) + (1 - dice_score(pred=preds, target=y_2d))
        self.logger.experiment.log_metric(f"Loss {stage}", loss, step=self.global_step)
        self.log("Train Loss", loss, on_step=True)

        # if True:
        #     self._log_step_figures(x, y_2d, preds, batch_idx)

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
            self._log_step_figures(x, y_2d, preds, batch_idx)

        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        return opt

    def _log_step_figures(self, x, y, y_hat, batch_idx):

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

        # cm = pl.metrics.functional.classification.confusion_matrix(y_hat.argmax(dim=1), y, normalize=False,
        #                                                                      num_classes=21)

        # if batch_idx == 0:
        #     self.logger.experiment.log_confusion_matrix(matrix=cm, file_name=f"confusion_matrix_{self.current_epoch}.json")

        plt.close("all")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0002, help="adam: learning rate")
        return parser
