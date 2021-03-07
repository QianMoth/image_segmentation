from tensorflow.keras.layers import *


def DoubleConv(filter_size, inputs):
    conv1 = Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    # BN1 = layers.BatchNormalization(axis=3)(conv1)
    conv2 = Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    BN2 = BatchNormalization(axis=3)(conv2)
    drop = Dropout(0.5)(BN2)
    return drop
