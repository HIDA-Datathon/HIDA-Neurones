# Number of loss functions from 
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Loss-Function-Reference-for-Keras-&-PyTorch
# For testing with simple UNETs
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Recall, Precision

from tensorflow.keras import backend as K

import tensorflow as tf


def FocalLoss(targets, inputs, alpha=0.8, gamma=2):    
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss


def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


#Keras
def IoULoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU


def FocalTverskyLoss(targets, inputs, alpha=0.5, beta=0.5, gamma=1, smooth=1e-6):
    
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = K.pow((1 - Tversky), gamma)
        
        return FocalTversky
    
    
def dice_coef_binary(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 2 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))


def dice_coef_binary_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_binary(y_true, y_pred)


# Add binary cross entropy
def BCE_dice(y_true, y_pred):
    cce = K.categorical_crossentropy(y_true[:, :, :], y_pred[:, :, :])
    dice = dice_loss(y_true[:, :, :], y_pred[:, :, :])
    dice2 = dice_coef_binary_loss(y_true[:, :, 1:], y_pred[:, :, 1:])
    return  0.33 * dice + 0.33 * cce  + 0.33 * dice2


def make_model(image_size, n_classes = 22, MODELPATH=None):
    inputs = Input(shape=(*image_size, 3), name='input_image')
   
    encoder = MobileNetV2(input_tensor=inputs, weights=MODELPATH, include_top=False, alpha=1.0)
    skip_connection_names = ["input_image", 
                             "block_1_expand_relu", 
                             "block_3_expand_relu", 
                             "block_6_expand_relu"]
    
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [32, 64, 128, 256]
    x = encoder_output
    
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

   # x = tf.keras.layers.Dropout(0.25)(x)
        
    x = Conv2D(n_classes, (1, 1), padding="same")(x)
    x = tf.keras.layers.Softmax(axis=-1)(x)
    
    model = Model(inputs, x)
    return model