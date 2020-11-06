from src.utils.Model import MyModel
import pytorch_lightning as pl
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

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
    image_path = "C:/Users/Tobias/Downloads/HIDA-ufz_image_challenge/photos_annotated/2019_0626_080354_004.jpg.jpg"
    file_index = image_path.index("/photos_annotated")
    file_end_index = image_path.index(".jpg")
    file_name = image_path[file_index:file_end_index]
    model = main()
    image = np.array(Image.open(image_path)) / 255
    image = np.moveaxis(image, -1, 0)
    image = np.expand_dims(image, 0)
    input_tensor = torch.from_numpy(image).float()

    output = model(input_tensor).argmax(dim=1).detach().cpu().numpy()
    
    #copy to all channels
    for i in range(3):
        output[i]=output[0]

    # save image as png
    save_path="data/HIDA-ufz_image_challenge/photos_predictions"
    im = Image.fromarray(output)
    im.save(os.path.join(save_path, file_name + "_pred.jpg.png"))
