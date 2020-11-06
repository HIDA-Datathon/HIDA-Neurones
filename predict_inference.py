from src.utils.Model import MyModel
import pytorch_lightning as pl
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob

def get_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--checkpoint_path', type=str, help="adam: learning rate")
    return parser

def main(args=None):

    pl.seed_everything(52)
    parser = ArgumentParser()

    script_args, _ = parser.parse_known_args(args)
    parser = get_args(parser)
    parser = MyModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)
    model = MyModel.load_from_checkpoint(**vars(args))
    return model


if __name__ == "__main__":
    IMAGE_SUFFIX = "*.jpg"
    data_dir="/home/robert/Neutrons_Net/data/HIDA-ufz_image_challenge/photos_annotated/"
    images = glob.glob(data_dir + "*.jpg")
    
    for path in images:
        file_index = path.index("photos_annotated")
        file_end_index = path.index(".jpg")
        file_name = path[file_index:file_end_index]
        model = main()
        image = np.array(Image.open(path)) / 255
        image = np.moveaxis(image, -1, 0)
        image = np.expand_dims(image, 0)
        input_tensor = torch.from_numpy(image).float()

        output = model(input_tensor).argmax(dim=1).detach().cpu().numpy()
    
        #copy to all channels
        output_3d = np.zeros((3,600,800))
        print(output_3d.shape)
        for i in range(2):
            output_3d[i]=output[0]
        output_3d=np.moveaxis(output_3d, 0, -1)
        # save image as png
        save_path="/home/robert/Neutrons_Net/data/predictions"
        im = Image.fromarray(np.uint8(output_3d))
        im.save(os.path.join(save_path, file_name + "_pred.jpg.png"))
