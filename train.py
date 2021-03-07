# 主函数
# 包括断点续存、训练好的模型参数提取、可视化
# coding = UTF-8

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import data
import numpy as np
from models.networks import myModel, Unet, BCDUnet

import tensorflow as tf
from tensorflow.keras import backend
import glob

# 加载数据
# 修改了文件夹的内容，文件夹内包含所有数据集。
train_images_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_Data/*jpg'
test_images_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Test_Data/*jpg'
train_masks_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_GroundTruth/*.png'
test_masks_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Test_GroundTruth/*.png'

# 将官方的训练集与测试集一起加载了
images_path = glob.glob(train_images_file_path) + glob.glob(test_images_file_path)
masks_path = glob.glob(train_masks_file_path) + glob.glob(test_masks_file_path)
# print(images_path)

train_count = int(len(images_path) * 0.8)  # 80%
test_count = len(images_path) - train_count  # 20%
print('数据集个数:', len(images_path),
      '训练集个数:', train_count, '测试集个数:', test_count)
# 数据集个数: 1279 训练集个数: 1023 测试集个数: 256


# 超参数
batch_size = 2
# 公司电脑只能 <= 2 # 最新的模型GPU已经不行了
# 学校电脑 <=25, 32不行其余没测试过
epochs = 10
buffer_size = train_count
steps_per_epoch = train_count // batch_size
validation_steps = 1
# ---------dice损失函数超参数----------
smooth = 1e-7  # 用于防止分母为0.

# 打乱数据集，重写
dataset = tf.data.Dataset.from_tensor_slices((images_path, masks_path))
dataset = dataset.map(data.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# 跳过test_count个
train = dataset.skip(test_count)
test = dataset.take(test_count)
# print(train, test)

train_dataset = train.cache().shuffle(buffer_size).batch(batch_size).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(batch_size)


# test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# print(train_dataset, test_dataset)

# train:  <SkipDataset shapes: ((256, 256, 3), (256, 256, 1)), types: (tf.float32, tf.float32)>
# test:   <TakeDataset shapes: ((256, 256, 3), (256, 256, 1)), types: (tf.float32, tf.float32)>
# train_dataset:  <PrefetchDataset shapes: ((None, 256, 256, 3), (None, 256, 256, 1)), types: (tf.float32, tf.float32)>
# test_dataset:   <BatchDataset shapes: ((None, 256, 256, 3), (None, 256, 256, 1)), types: (tf.float32, tf.float32)>


# # 看一下数据加载是否正常
# sample_image, sample_mask = [], []
# for image, mask in train.take(1):
#     sample_image, sample_mask = image, mask
# show.display([sample_image, sample_mask])

# # 数据增强,有待商榷
# train_dataset = train_dataset.reshape(train_dataset.shape[0], 256, 256, 3)
# image_gen_train = ImageDataGenerator(
#     rescale=1. / 1.,  # 归一化
#     rotation_range=45,  # 随机90°旋转
#     width_shift_range=.15,  # 宽度偏移
#     height_shift_range=.15,
#     horizontal_flip=False,
#     zoom_range=0.5)  # 随机缩放阈量50%
# image_gen_train.fit(train_dataset)


# -------------dice损失函数-------------
# smooth = 1e-7  # 用于防止分母为0.


def dice_coef(y_true, y_pred):
    y_true_f = backend.flatten(y_true)  # 将 y_true 拉伸为一维.
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f * y_true_f) + backend.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


# ------------------------------------

# output_channels = 2

# 加载模型, 这样是为了在多个模型的情况下便于修改
# model = myModel.QianNet
# model = Unet.Unet
model = BCDUnet.BCDU_net(input_size=(256, 256, 3))

# 配置训练方法
model.compile(optimizer='adam',
              loss=dice_coef_loss,
              metrics=["binary_accuracy", dice_coef])

# 断点续训
checkpoint_save_path = "./data/checkpoint/MyModel.ckpt"

# class DisplayCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         # clear_output(wait=True)  # 每次输出图像前关闭上一次
#         show.show_predictions(checkpoint_save_path, test_dataset)
#         # print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


# 训练模型 问题batch size打开会报错
history = model.fit(
    # image_gen_train.flow(train_dataset, batch_size=batch_size),
    train_dataset,
    # batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_dataset,
    validation_steps=validation_steps,
    # callbacks=[]
)

# # 不同图片展示
# show.show_predictions(checkpoint_save_path, test_dataset, num=5)

# 模型结构展示
model.summary()

# 可视化
# 显示train和test的acc和loss曲线
acc = history.history['dice_coef']
val_acc = history.history['val_dice_coef']
loss = history.history['loss']
val_loss = history.history['val_loss']

# show.acc_loss(acc, val_acc, loss, val_loss)
