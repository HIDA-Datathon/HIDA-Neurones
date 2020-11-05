from src.utils.DataLoader import NeutronDataLoader
from src.utils.Model import MyModel
import pytorch_lightning as pl
from torchvision.transforms import CenterCrop


def main():

    dm = NeutronDataLoader(batch_size=4,
                           data_dir="/gpfs/home/machnitz/HIDA/HIDA-ufz_image_challenge/photos_annotated")
    model = MyModel()
    trainer = pl.Trainer(max_epochs=20, gpus=1)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
