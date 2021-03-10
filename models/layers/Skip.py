import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def AttentionGate(channel, x, g):
    theta_att = Conv2D(channel, 1, strides=1, padding='same', use_bias=True)(x)
    # theta_att = BatchNormalization(axis=3)(theta_att)
    # theta_att = LayerNormalization(axis=3)(theta_att)

    phi_g = Conv2D(channel, 1, strides=1, padding='same', use_bias=True)(g)
    # phi_g = BatchNormalization(axis=3)(phi_g)
    # phi_g = LayerNormalization(axis=3)(phi_g)

    query = add([theta_att, phi_g])
    # f = ReLU(inplace=True)(query)
    f = ReLU(True)(query)

    psi_f = Conv2D(1, 1, strides=1, padding='same', use_bias=True, activation='sigmoid')(f)
    X_att = BatchNormalization(axis=3)(psi_f)
    # X_att = LayerNormalization(axis=3)(psi_f)
    X_att = multiply([x, X_att])

    return X_att


def BConvLSTMSkip(filter_size, inputs, skips):
    size = np.int32((256 * 64) / filter_size)
    x1 = Reshape(target_shape=(1, size, size, filter_size))(inputs)
    x2 = Reshape(target_shape=(1, size, size, filter_size))(skips)
    merge = concatenate([x1, x2], axis=1)

    merge = ConvLSTM2D(filters=filter_size // 2, kernel_size=(3, 3), padding='same', return_sequences=False,
                       go_backwards=True, kernel_initializer='he_normal')(merge)
    return merge
