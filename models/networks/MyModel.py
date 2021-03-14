from tensorflow.keras import Model
from tensorflow.keras.layers import *
from models.layers.Conv import DoubleConv
from models.layers.Skip import BConvLSTMSkip, AttentionGate


def BSCUNet_Qian(input_size=(256, 256, 1)):
    inputs = Input(shape=input_size)
    conv1 = DoubleConv(64, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = DoubleConv(128, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = DoubleConv(256, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # D1
    conv4_1 = DoubleConv(512, pool3)
    # D2
    conv4_2 = DoubleConv(512, conv4_1)
    # D3
    merge_dense = concatenate([conv4_2, conv4_1], axis=3)
    conv4_3 = DoubleConv(512, merge_dense)

    up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal',
                          activation='relu')(conv4_3)
    BN3 = BatchNormalization(axis=3)(up3)
    skip3 = BConvLSTMSkip(256, BN3, conv3)
    conv3 = DoubleConv(256, skip3)

    up2 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal',
                          activation='relu')(conv3)
    BN2 = BatchNormalization(axis=3)(up2)
    skip2 = BConvLSTMSkip(128, BN2, conv2)
    conv2 = DoubleConv(128, skip2)

    up1 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal',
                          activation='relu')(conv2)
    BN1 = BatchNormalization(axis=3)(up1)
    skip1 = BConvLSTMSkip(64, BN1, conv1)
    conv1 = DoubleConv(64, skip1)

    conv1 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Conv2D(1, 1, activation='sigmoid')(conv1)

    model = Model(inputs, conv1)
    return model


def AttentionDenseUNet(input_size=(256, 256, 1)):
    # print("=========encoder starte========")
    inputs = Input(shape=input_size)
    # print("inputs: ", inputs)
    conv1 = DoubleConv(64, inputs)
    # print("conv1: ", conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print("pool1: ", pool1)
    conv2 = DoubleConv(128, pool1)
    # print("conv2: ", conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print("pool2: ", pool2)
    conv3 = DoubleConv(256, pool2)
    # print("conv3: ", conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # print("pool3: ", pool3)
    # print("=========encoder end========")

    # D1
    conv4_1 = DoubleConv(512, pool3)
    # print("drop4_1: ", conv4_1)
    # D2
    conv4_2 = DoubleConv(512, conv4_1)
    # print("drop4_2: ", conv4_2)
    # D3
    merge_dense = concatenate([conv4_2, conv4_1], axis=3)
    conv4_3 = DoubleConv(512, merge_dense)
    # print("drop4_3: ", conv4_3)

    # print("=========decoder start========")
    up5 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv4_3)
    # print("up5: ", up5)
    skip5 = AttentionGate(256, up5, conv3)
    # print("skip8: ", skip5)
    conv5 = DoubleConv(256, skip5)
    # print("conv8: ", conv5)

    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv5)
    # print("up6: ", up6)
    skip6 = AttentionGate(128, up6, conv2)
    # print("skip6: ", skip6)
    conv6 = DoubleConv(128, skip6)
    # print("conv6: ", conv6)

    up7 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
    # print("up7: ", up7)
    skip7 = AttentionGate(64, up7, conv1)
    # print("skip7: ", skip7)
    conv7 = DoubleConv(64, skip7)
    # print("conv7: ", conv7)
    # print("=========decoder end========")

    conv8 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    # print("conv9: ", conv8)
    conv9 = Conv2D(1, 1, activation='sigmoid')(conv8)
    # print("conv10: ", conv9)

    model = Model(inputs, conv9)
    return model
