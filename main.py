# 主函数
# 包括断点续存、训练好的模型参数提取、可视化
# coding = UTF-8

import data
from networks import MyModel
import show

import tensorflow as tf
from tensorflow.keras import callbacks, backend
import glob
import os

# 加载数据
# 修改了文件夹的内容，文件夹内包含所有数据集。
images_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_Data/*jpg'
masks_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_GroundTruth/*.png'

images_path = glob.glob(images_file_path)
masks_path = glob.glob(masks_file_path)
# print(images_path)

train_count = int(len(images_path) * 0.8)  # 80%
test_count = len(images_path) - train_count  # 20%
print('数据集个数:', len(images_path),
      '训练集个数:', train_count, '测试集个数:', test_count)
# 数据集个数: 1279 训练集个数: 1023 测试集个数: 256

# 超参数
batch_size = 25
# 公司电脑只能 <= 2
# 学校电脑 <=25, 32不行其余没测试过
epochs = 15
buffer_size = train_count
steps_per_epoch = train_count // batch_size
validation_steps = 1
# ---------dice损失函数超参数----------
smooth = 1e-7  # 用于防止分母为0.

train_dataset, test_dataset, train, test = data.distribute(images_path, masks_path, test_count, batch_size, buffer_size)
# train:  <SkipDataset shapes: ((256, 256, 3), (256, 256, 1)), types: (tf.float32, tf.float32)>
# test:   <TakeDataset shapes: ((256, 256, 3), (256, 256, 1)), types: (tf.float32, tf.float32)>
# train_dataset:  <PrefetchDataset shapes: ((None, 256, 256, 3), (None, 256, 256, 1)), types: (tf.float32, tf.float32)>
# test_dataset:   <BatchDataset shapes: ((None, 256, 256, 3), (None, 256, 256, 1)), types: (tf.float32, tf.float32)>

# 看一下数据加载是否正常
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

# output_channels = 2

# 加载模型
model = MyModel


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

# 配置训练方法
model.compile(optimizer='adam',
              loss=dice_coef_loss,
              metrics=["binary_accuracy", dice_coef])

# # 断点续训
# checkpoint_save_path = "./data/checkpoint/MyModel.ckpt"  # 导出数据路径
# # 读取模型，判断是否存在文件
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------下载模型-------------')
#     model.load_weights(checkpoint_save_path)
# # 保存模型
# cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                         save_weights_only=True,
#                                         save_best_only=True)
#
#
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
    # callbacks=[cp_callback, DisplayCallback()]
)

# 不同图片展示
# show.show_predictions(checkpoint_save_path, test_dataset, num=5)

# 模型结构展示
model.summary()

# 可视化
# 显示train和test的acc和loss曲线
acc = history.history['dice_coef']
val_acc = history.history['val_dice_coef']
loss = history.history['loss']
val_loss = history.history['val_loss']

show.acc_loss(acc, val_acc, loss, val_loss)
