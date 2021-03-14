from tensorflow.keras import Model
from tensorflow.keras.layers import *
from models.layers.Conv import DoubleConv
from models.layers.Skip import BConvLSTMSkip, AttentionGate


def ADUNet_L5(input_size=(256, 256, 1)):
    inputs = Input(shape=input_size)
    conv1 = DoubleConv(64, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = DoubleConv(128, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = DoubleConv(256, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = DoubleConv(512, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # D1
    conv_bottom_1 = DoubleConv(1024, pool4)
    # D2
    conv_bottom_2 = DoubleConv(1024, conv_bottom_1)
    # D3
    merge_dense = concatenate([conv_bottom_2, conv_bottom_1], axis=3)
    conv_bottom_3 = DoubleConv(1024, merge_dense)

    up4 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv_bottom_3)
    skip4 = AttentionGate(512, up4, conv4)
    conv4 = DoubleConv(512, skip4)

    up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv4)
    skip3 = AttentionGate(256, up3, conv3)
    conv3 = DoubleConv(256, skip3)

    up2 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv3)
    skip2 = AttentionGate(128, up2, conv2)
    conv2 = DoubleConv(128, skip2)

    up1 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv2)
    skip1 = AttentionGate(64, up1, conv1)
    conv1 = DoubleConv(64, skip1)

    conv_output = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv_output = Conv2D(1, 1, activation='sigmoid')(conv_output)
    # conv_output = Conv2D(1, 1, activation='softmax')(conv_output)

    model = Model(inputs, conv_output)
    return model


def ADUNet_L4(input_size=(256, 256, 1)):
    inputs = Input(shape=input_size)
    conv1 = DoubleConv(64, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = DoubleConv(128, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = DoubleConv(256, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # D1
    conv_bottom_1 = DoubleConv(512, pool3)
    # D2
    conv_bottom_2 = DoubleConv(512, conv_bottom_1)
    # D3
    merge_dense = concatenate([conv_bottom_2, conv_bottom_1], axis=3)
    conv_bottom_3 = DoubleConv(512, merge_dense)

    up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv_bottom_3)
    skip3 = AttentionGate(256, up3, conv3)
    conv3 = DoubleConv(256, skip3)

    up2 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv3)
    skip2 = AttentionGate(128, up2, conv2)
    conv2 = DoubleConv(128, skip2)

    up1 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv2)
    skip1 = AttentionGate(64, up1, conv1)
    conv1 = DoubleConv(64, skip1)

    conv_output = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv_output = Conv2D(1, 1, activation='sigmoid')(conv_output)
    # conv_output = Conv2D(1, 1, activation='softmax')(conv_output)

    model = Model(inputs, conv_output)
    return model


def AttentionUNet(input_size=(256, 256, 1)):
    inputs = Input(shape=input_size)
    conv1 = DoubleConv(64, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = DoubleConv(128, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = DoubleConv(256, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv_bottom = DoubleConv(512, pool3)

    up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv_bottom)
    skip3 = AttentionGate(256, up3, conv3)
    conv3 = DoubleConv(256, skip3)

    up2 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv3)
    skip2 = AttentionGate(128, up2, conv2)
    conv2 = DoubleConv(128, skip2)

    up1 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv2)
    skip1 = AttentionGate(64, up1, conv1)
    conv1 = DoubleConv(64, skip1)

    conv_output = Conv2D(1, 1, activation='sigmoid')(conv1)

    model = Model(inputs, conv_output)
    return model


def UNet(input_size=(256, 256, 1)):
    inputs = Input(shape=input_size)
    conv1 = DoubleConv(64, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = DoubleConv(128, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = DoubleConv(256, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = DoubleConv(512, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv_bottom = DoubleConv(1024, pool4)

    up4 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv_bottom)
    skip4 = concatenate([up4, conv4], axis=3)
    conv4 = DoubleConv(512, skip4)

    up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv4)
    skip3 = concatenate([up3, conv3], axis=3)
    conv3 = DoubleConv(256, skip3)

    up2 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv3)
    skip2 = concatenate([up2, conv2], axis=3)
    conv2 = DoubleConv(128, skip2)

    up1 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv2)
    skip1 = concatenate([up1, conv1], axis=3)
    conv1 = DoubleConv(64, skip1)

    conv_output = Conv2D(1, 1, activation='sigmoid')(conv1)

    model = Model(inputs, conv_output)
    return model


def BSCUNet(input_size=(256, 256, 1)):
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
