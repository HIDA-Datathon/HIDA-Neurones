from src.utils.DataLoader import NeutronDataLoader
from src.utils.Model import MyModel
import pytorch_lightning as pl
from torchvision.transforms import CenterCrop
from pytorch_lightning import loggers as pl_loggers
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

EXPERIMENT_NAME = "SegResnet-20 inference"
CHECKPOINT = "/gpfs/home/machnitz/HIDA/HIDA-Neurones/comet_logs/HIDA/8b6b8eb333ed4e08a6544d788680b6ea/checkpoints/epoch=19.ckpt"
# CHECKPOINT = "C:/Users\\Tobias\\PycharmProjects\\HIDA-Neurones\\comet_logs\\HIDA\\decbc39aac0c432a811e6dc61d45e807\\checkpoints\\epoch=0.ckpt"


def main(args=None):
    pl.seed_everything(52)

    parser = ArgumentParser()
    dm_cls = NeutronDataLoader

    script_args, _ = parser.parse_known_args(args)
    parser = dm_cls.add_argparse_args(parser)
    parser = MyModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    comet_logger = pl_loggers.CometLogger(save_dir="comet_logs", experiment_name=EXPERIMENT_NAME,
                                          project_name="HIDA", offline=True)

    dm = dm_cls.from_argparse_args(args)
    dm.setup()
    model = MyModel.load_from_checkpoint(CHECKPOINT, **vars(args))

    trainer = pl.Trainer.from_argparse_args(args, logger=comet_logger)
    trainer.test(model, test_dataloaders=dm.test_dataloader())


if __name__ == "__main__":
    main()
