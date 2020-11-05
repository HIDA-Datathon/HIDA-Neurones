from src.utils.DataLoader import NeutronDataLoader
from src.utils.Model import MyModel
import pytorch_lightning as pl
from torchvision.transforms import CenterCrop
from pytorch_lightning import loggers as pl_loggers
from argparse import ArgumentParser


EXPERIMENT_NAME = "Training"


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
    dm = dm_cls.from_argparse_args(args,
                           data_dir="/gpfs/home/machnitz/HIDA/HIDA-ufz_image_challenge/photos_annotated"
                           )
    model = MyModel(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, logger=comet_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
