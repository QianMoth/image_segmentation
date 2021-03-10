from tensorflow.keras import Model
from tensorflow.keras.layers import *
from models.layers.Conv import DoubleConv
from models.layers.Skip import BConvLSTMSkip, AttentionGate


def BSCUNet_Qian(input_size=(256, 256, 1)):
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

    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal',
                          activation='relu')(drop4_3)
    BN6 = BatchNormalization(axis=3)(up6)
    skip6 = BConvLSTMSkip(256, BN6, conv3)
    conv6 = DoubleConv(256, skip6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal',
                          activation='relu')(conv6)
    BN7 = BatchNormalization(axis=3)(up7)
    skip7 = BConvLSTMSkip(128, BN7, conv3)
    conv7 = DoubleConv(128, skip7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal',
                          activation='relu')(conv7)
    BN8 = BatchNormalization(axis=3)(up8)
    skip8 = BConvLSTMSkip(64, BN8, conv3)
    conv8 = DoubleConv(64, skip8)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)
    return model


def AttentionUNet_Qian(input_size=(256, 256, 1)):
    print("=========encoder starte========")
    inputs = Input(input_size)
    print("inputs: ", inputs)
    conv1 = DoubleConv(64, inputs)
    print("conv1: ", conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1: ", pool1)
    conv2 = DoubleConv(128, pool1)
    print("conv2: ", conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2: ", pool2)
    conv3 = DoubleConv(256, pool2)
    print("conv3: ", conv3)
    drop3 = Dropout(0.5)(conv3)
    print("drop3: ", drop3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    print("pool3: ", pool3)
    print("=========encoder end========")

    # D1
    conv4_1 = DoubleConv(512, pool3)
    drop4_1 = Dropout(0.5)(conv4_1)
    print("drop4_1: ", drop4_1)
    # D2
    conv4_2 = DoubleConv(512, drop4_1)
    drop4_2 = Dropout(0.5)(conv4_2)
    print("drop4_2: ", drop4_2)
    # D3
    merge_dense = concatenate([drop4_2, drop4_1], axis=3)
    conv4_3 = DoubleConv(512, merge_dense)
    drop4_3 = Dropout(0.5)(conv4_3)
    print("drop4_3: ", drop4_3)

    print("=========decoder start========")
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(drop4_3)
    print("up6: ", up6)
    skip6 = AttentionGate(256, up6, conv3)
    print("skip6: ", skip6)
    conv6 = DoubleConv(256, skip6)
    print("conv6: ", conv6)

    up7 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
    print("up7: ", up7)
    skip7 = AttentionGate(128, up7, conv2)
    print("skip7: ", skip7)
    conv7 = DoubleConv(128, skip7)
    print("conv7: ", conv7)

    up8 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv7)
    print("up8: ", up8)
    skip8 = AttentionGate(64, up8, conv1)
    print("skip8: ", skip8)
    conv8 = DoubleConv(64, skip8)
    print("conv8: ", conv8)
    print("=========decoder end========")

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    print("conv9: ", conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    print("conv10: ", conv10)

    model = Model(inputs, conv10)
    return model
