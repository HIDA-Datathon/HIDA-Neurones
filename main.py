from src.utils.DataLoader import NeutronDataLoader
from src.utils.Model import MyModel
import pytorch_lightning as pl
from torchvision.transforms import CenterCrop


def main():

    dm = NeutronDataLoader(batch_size=2, transform=CenterCrop(100))
    model = MyModel()
    trainer = pl.Trainer(fast_dev_run=True, gpus=1)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
