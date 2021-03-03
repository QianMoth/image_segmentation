# https://hansy.tech/2020/07/1919729354.html
import tensorflow as tf
from tensorflow.keras import layers, Model


class DoubleConv(layers.Layer):
    def __init__(self, filter_size):
        super(DoubleConv, self).__init__()
        self.conv1 = layers.Conv2D(filter_size, 3, padding='same', activation='relu')
        # self.BN1 = layers.BatchNormalization(axis=3)
        self.conv2 = layers.Conv2D(filter_size, 3, padding='same', activation='relu')
        self.BN2 = layers.BatchNormalization(axis=3)
        self.drop = layers.Dropout(0.2)

    def call(self, inputs, **kwargs):
        l = self.conv1(inputs)
        # l = self.BN1(l)
        l = self.conv2(l)
        l = self.BN2(l)
        l = self.drop(l)
        return l


# Attention输入两个值，输出一个值
class AttentionBlock(layers.Layer):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = tf.keras.Sequential(
            layers.Conv2D(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            layers.BatchNormalization(F_int)
        )

        self.W_x = tf.keras.Sequential(
            layers.Conv2D(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            layers.BatchNormalization(F_int)
        )

        self.psi = tf.keras.Sequential(
            layers.Conv2D(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True, activation='sigmoid'),
            layers.BatchNormalization(1),
        )

        self.relu = layers.ReLU(inplace=True)

    def call(self, g, x, **kwargs):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(layers.concatenate([g1, x1], axis=3))
        # psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(Model):
    def __init__(self, output_channels):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(64)
        self.pool1 = layers.MaxPool2D((2, 2), strides=2, padding='same')
        self.down2 = DoubleConv(128)
        self.pool2 = layers.MaxPool2D((2, 2), strides=2, padding='same')
        self.down3 = DoubleConv(256)
        self.pool3 = layers.MaxPool2D((2, 2), strides=2, padding='same')
        self.down4 = DoubleConv(512)
        self.pool4 = layers.MaxPool2D((2, 2), strides=2, padding='same')

        self.bottom = DoubleConv(1024)

        # self.up_sample4 = layers.UpSampling2D((2, 2), interpolation='nearest')
        self.up_sample4 = layers.Conv2DTranspose(512, (2, 2), strides=2, padding='same')
        self.up4 = DoubleConv(512)
        self.up_sample3 = layers.Conv2DTranspose(256, (2, 2), strides=2, padding='same')
        self.up3 = DoubleConv(256)
        self.up_sample2 = layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')
        self.up2 = DoubleConv(128)
        self.up_sample1 = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')
        self.up1 = DoubleConv(64)
        # self.Up2 = up_conv(ch_in=128, ch_out=64)
        # self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        # self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        d1 = self.down1(inputs)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        l1 = self.bottom(self.pool4(d4))

        u4 = self.up4(layers.concatenate([self.up_sample4(l1), d4], axis=3))
        u3 = self.up3(layers.concatenate([self.up_sample3(u4), d3], axis=3))
        u2 = self.up2(layers.concatenate([self.up_sample2(u3), d2], axis=3))
        u1 = self.up1(layers.concatenate([self.up_sample1(u2), d1], axis=3))
        # d2 = self.Up2(d3)
        # x1 = self.Att2(g=d2, x=x1)
        # d2 = torch.cat((x1, d2), dim=1)
        # d2 = self.Up_conv2(d2)

        outputs = self.conv10(u1)

        return outputs


# def unet():
#     return UNet(output_channels=2)
MyModel = UNet(output_channels=2)

# 问题1：output_channels=2
# 加attention
# 把条跳跃连接反向，做个负反馈怎么样？
