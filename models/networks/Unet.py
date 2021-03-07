import tensorflow as tf
from tensorflow.keras import layers, Model


# 如果无特定必要，尽可能避免使用Model子类化的方式构建模型，这种方式提供了极大的灵活性，但也有更大的概率出错。
class DoubleConv(layers.Layer):
    def __init__(self, filter_size):
        super(DoubleConv, self).__init__()
        self.conv1 = layers.Conv2D(filter_size, 3, padding='same', activation='relu')
        # self.BN1 = layers.BatchNormalization(axis=3)
        self.conv2 = layers.Conv2D(filter_size, 3, padding='same', activation='relu')
        self.BN2 = layers.BatchNormalization(axis=3)
        self.drop = layers.Dropout(0.5)

    def call(self, inputs, training=None, **kwargs):
        l = self.conv1(inputs)
        # l = self.BN1(l)
        l = self.conv2(l)
        if training:
            l = self.BN2(l)
            l = self.drop(l)
        return l


class Unet(Model):
    def __init__(self, output_channels):
        super(Unet, self).__init__()
        self.down1 = DoubleConv(64)
        self.pool1 = layers.MaxPool2D((2, 2), strides=2, padding='same')
        self.down2 = DoubleConv(128)
        self.pool2 = layers.MaxPool2D((2, 2), strides=2, padding='same')
        self.down3 = DoubleConv(256)
        self.pool3 = layers.MaxPool2D((2, 2), strides=2, padding='same')
        self.down4 = DoubleConv(512)
        self.pool4 = layers.MaxPool2D((2, 2), strides=2, padding='same')

        self.bottom1 = DoubleConv(1024)

        self.up_sample4 = layers.Conv2DTranspose(512, (2, 2), strides=2, padding='same')
        self.up4 = DoubleConv(512)
        self.up_sample3 = layers.Conv2DTranspose(256, (2, 2), strides=2, padding='same')
        self.up3 = DoubleConv(256)
        self.up_sample2 = layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')
        self.up2 = DoubleConv(128)
        self.up_sample1 = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')
        self.up1 = DoubleConv(64)

        self.conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        d1 = self.down1(inputs)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        l1 = self.bottom1(self.pool4(d4))

        u4 = self.up4(layers.concatenate([self.up_sample4(l1), d4], axis=3))
        u3 = self.up3(layers.concatenate([self.up_sample3(u4), d3], axis=3))
        u2 = self.up2(layers.concatenate([self.up_sample2(u3), d2], axis=3))
        u1 = self.up1(layers.concatenate([self.up_sample1(u2), d1], axis=3))

        outputs = self.conv10(u1)

        return outputs


Unet = Unet(output_channels=2)
