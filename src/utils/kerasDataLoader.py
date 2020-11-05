from tensorflow import keras
import tensorflow as tf
from glob import glob
import os
import warnings
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


class DataGenerator(keras.utils.Sequence):

    def __init__(self, datapath='.././data/HIDA-ufz_image_challenge/photos_annotated', 
                       image_size=(224, 224), batch_size=1, shuffle=False, step='train',
                nb_labels=20, augmentation=None):
        """Data generator for validation and training.
        Args:
            image_csv (pd.DataFrame): Dataframe with absolute path to images in
                                      "Filename", and bounding boxe locations.
            image_size (tuple, optional): Target size of images. Defaults to (224, 224).
            batch_size (int, optional): Batch size. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle before creating batches. Defaults to False.
        """
        self.datapath = datapath
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.image_list = sorted(glob(os.path.join(self.datapath, '*.jpg')))
        self.label_list = sorted(glob(os.path.join(self.datapath, '*.png')))
        self.step = step
        self.nb_labels = nb_labels
        self.augmentation=augmentation
        
        if len(self.image_list) != len(self.label_list):
            warnings.warn('Warning, data length is different!')
            
        self.data_size = len(self.image_list)
        
        if self.step != 'train' and self.shuffle:
            warning.warn('Warning, shuffle is True, but it is not a training generator!')
            
        
        if self.step == 'train':
            self.image_list = self.image_list[: int(0.8 * self.data_size)]
            self.label_list = self.label_list[: int(0.8 * self.data_size)]
        elif self.step == 'valid':
            self.image_list = self.image_list[int(0.8 * self.data_size):int(0.9 * self.data_size)]
            self.label_list = self.label_list[int(0.8 * self.data_size):int(0.9 * self.data_size)]
        elif self.step == 'test':
            self.image_list = self.image_list[int(0.9 * self.data_size):]
            self.label_list = self.label_list[int(0.9 * self.data_size):]
            
            
        self.set_size = len(self.image_list)
        
        self.images, self.labels = self.load_data()
        
        
        self.on_epoch_end()
        
    def load_data(self):

        image_array = []
        label_array = []
        
        
        for img, label in tqdm(zip(self.image_list, self.label_list), total=self.set_size, desc='Loading images'):
            if self._assert_images(img, label):
                image_array.append(np.array(Image.open(img).resize(self.image_size)))
                
                label = np.array(Image.open(label).resize(self.image_size))
                if len(label.shape) == 2:
                    label_array.append(label)
                elif len(label.shape) == 3:
                    label_array.append(label[:, :, 0])
                else:
                    print("Error")
    
        return np.array(image_array), np.array(label_array)
        

    def _assert_images(self, im_path, label_path):
        
        if im_path[0].split('/')[-1].split('.')[0] == label_path[0].split('/')[-1].split('.')[0]:
            return True
        else:
            warnings.warn("Warning, data not matching!")
            return False


    def __len__(self):
        # Calculate the lenght of the generator (i.e. number of batches in epoch)
        return int(np.floor(self.set_size / self.batch_size))

    
    def on_epoch_end(self):
        # Indices = number of files in there (even if it's df)
        self.indexes = np.arange(self.set_size)

        if self.shuffle:
            np.random.shuffle(self.indexes)
        

    def __getitem__(self, index):
        # Create indices (shuffled)
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, indexes):
        X, y = [], []

        # Generate data
        for idx in indexes:
            # Load image, save dimensions and resize:
            img = self.images[idx]
            mask = self.labels[idx]
            
            
            if self.augmentation:
                augmented = self.augmentation(self.image_size)(image=img, mask=mask)
                
                image_augm = augmented['image']
                mask_augm = augmented['mask'].reshape(*self.image_size, self.nb_labels)
                
                # divide by 255 to normalize images from 0 to 1
                img = image_augm/255
                mask = mask_augm
            
            X.append(img)
            y.append(mask)

        return np.array(X), np.array(y)