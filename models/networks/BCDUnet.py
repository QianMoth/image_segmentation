from __future__ import division
import numpy as np
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
import tensorflow as tf


def DoubleConv(filter_size, inputs):
    conv1 = layers.Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    # BN1 = layers.BatchNormalization(axis=3)(conv1)
    conv2 = layers.Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    BN2 = layers.BatchNormalization(axis=3)(conv2)
    drop = layers.Dropout(0.5)(BN2)
    return drop


def LSTMSkip(filter_size, inputs, skips):
    BN = BatchNormalization(axis=3)(inputs)
    relu = Activation('relu')(BN)

    size = np.int32((256 * 64) / filter_size)
    x1 = Reshape(target_shape=(1, size, size, filter_size))(relu)
    x2 = Reshape(target_shape=(1, size, size, filter_size))(skips)
    merge = concatenate([x1, x2], axis=1)
    merge = ConvLSTM2D(filters=filter_size // 2, kernel_size=(3, 3), padding='same', return_sequences=False,
                       go_backwards=True, kernel_initializer='he_normal')(merge)

    return merge


def BCDU_net(input_size=(256, 256, 1)):
    # N = input_size[0]
    inputs = Input(input_size)
    conv1 = DoubleConv(64, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = DoubleConv(128, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = DoubleConv(256, pool2)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    # D1
    conv4_1 = DoubleConv(512, pool3)
    drop4_1 = Dropout(0.5)(conv4_1)
    # D2
    conv4_2 = DoubleConv(512, drop4_1)
    drop4_2 = Dropout(0.5)(conv4_2)
    # D3
    merge_dense = concatenate([drop4_2, drop4_1], axis=3)
    conv4_3 = DoubleConv(512, merge_dense)
    drop4_3 = Dropout(0.5)(conv4_3)

    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(drop4_3)
    skip6 = LSTMSkip(256, up6, conv3)
    conv6 = DoubleConv(256, skip6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
    skip7 = LSTMSkip(128, up7, conv2)
    conv7 = DoubleConv(128, skip7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv7)
    skip8 = LSTMSkip(64, up8, conv1)
    conv8 = DoubleConv(64, skip8)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)
    return model


def Unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop4))
    merge6 = concatenate([drop3, up6], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv2, up7], axis=3)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv1, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation='sigmoid')(conv8)

    model = Model(inputs, conv9)
    return model
