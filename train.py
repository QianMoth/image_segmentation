# coding = UTF-8

import os
import data
import glob
import numpy as np
from models.networks import MyModel, Unet, BCDUnet
from models.losses import *

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import metrics, optimizers, losses
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# 可视化
import show
from random import randint
from IPython.display import SVG
import IPython.display as display
from IPython.display import clear_output
from tensorflow.keras.utils import plot_model, model_to_dot
import matplotlib.pyplot as plt

print(tf.version.VERSION)

# 使用CPU计算
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# save path
model_name = 'ADUNet_Qian'
save_model = model_name + '.h5'
save_dir = os.path.join(os.getcwd(), 'output')
save_model_path = os.path.join(save_dir, save_model)

# load datasets
# 修改了文件夹的内容，文件夹内包含所有数据集。
train_images_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_Data/*jpg'
test_images_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Test_Data/*jpg'
train_masks_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_GroundTruth/*.png'
test_masks_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Test_GroundTruth/*.png'

# 将官方的训练集与测试集一起加载了
# images_path = glob.glob(train_images_file_path) + glob.glob(test_images_file_path)
# masks_path = glob.glob(train_masks_file_path) + glob.glob(test_masks_file_path)
# 测试用
images_path = glob.glob(test_images_file_path)
masks_path = glob.glob(test_masks_file_path)
# print(images_path)

train_count = int(len(images_path) * 0.03)  # 80%
test_count = int((len(images_path) - train_count) * 0.01)  # 20%
print('数据集个数:', len(images_path),
      '训练集个数:', train_count, '测试集个数:', test_count)
# 数据集个数: 1279 训练集个数: 1023 测试集个数: 256


# 超参数
BATCH_SIZE = 1
# 公司电脑只能 <= 2 # 最新的模型GPU已经不行了
# 学校电脑 <=25, 32不行其余没测试过
EPOCHS = 3
BUFFER_SIZE = train_count
STEPS_PER_EPOCH = train_count // BATCH_SIZE
VALIDATION_STEPS = 1

# 打乱数据集，重写
dataset = tf.data.Dataset.from_tensor_slices((images_path, masks_path))
dataset = dataset.map(data.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# 跳过test_count个
train = dataset.skip(test_count)
test = dataset.take(test_count)
# print(train, test)
train_dataset = train.batch(BATCH_SIZE).shuffle(BUFFER_SIZE).cache().repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


# train:  <SkipDataset shapes: ((256, 256, 3), (256, 256, 1)), types: (tf.float32, tf.float32)>
# test:   <TakeDataset shapes: ((256, 256, 3), (256, 256, 1)), types: (tf.float32, tf.float32)>
# train_dataset:  <PrefetchDataset shapes: ((None, 256, 256, 3), (None, 256, 256, 1)), types: (tf.float32, tf.float32)>
# test_dataset:   <BatchDataset shapes: ((None, 256, 256, 3), (None, 256, 256, 1)), types: (tf.float32, tf.float32)>

for i, _ in enumerate(test_dataset.element_spec):
    test_dataset_i = test_dataset.map(lambda *args: args[i]).map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter(f'output/mydata_{i}.tfrecord')
    writer.write(test_dataset_i)

# 看一下数据加载是否正常
sample_image, sample_mask = [], []
for image, mask in train.take(randint(0, train_count)):
    sample_image, sample_mask = image, mask
show.display_sample([sample_image, sample_mask])

# #　模型加载
# checkpoint_save_path = 'output/ADUNet_Qian.h5'
# if os.path.exists(checkpoint_save_path):
#     print('-------------下载模型-------------')
#     model = models.load_model(checkpoint_save_path,
#                               custom_objects={'dice_coef_loss': dice_coef_loss,
#                                               'dice_coef': dice_coef})

# 加载模型, 这样是为了在多个模型的情况下便于修改
model = MyModel.AttentionDenseUNet(input_size=(256, 256, 3))
# 模型结构
plot_model(model, to_file=os.path.join(save_dir, model_name + '_model.png'))

# build
# 评价函数和 损失函数 相似，只不过评价函数的结果不会用于训练过程中。
model.compile(loss=dice_coef_loss,
              optimizer=Adam(learning_rate=1e-3),
              metrics=[metrics.binary_accuracy, dice_coef])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)  # 在验证集的误差不再下降时，中断训练
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)  # 当标准评估停止提升时，降低学习速率。
cp_callback = ModelCheckpoint(save_model_path, save_best_only=True)  # 保存模型


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # clear_output(wait=True)  # 每次输出图像前关闭上一次
        show.show_predictions(save_model_path, test_dataset)
        # print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


# 训练模型
history = model.fit(train_dataset, epochs=EPOCHS,

                    # shuffle=True,  # 则训练数据将在每个 epoch 混洗, 验证集永远不会混洗
                    # validation_split=0.2,  # 使用的验证数据将是最后 20％ 的数据。

                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_dataset,
                    validation_steps=VALIDATION_STEPS,

                    callbacks=[early_stopping,
                               DisplayCallback(),
                               cp_callback,
                               reduce_lr],
                    verbose=1)

# # 模型结构展示
# model.summary()

# # 不同图片展示
# show.show_predictions(save_model_path, test_dataset, num=2)

# 可视化
print(history.history.keys())
acc = history.history['dice_coef']
val_acc = history.history['val_dice_coef']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 绘制训练 & 验证的准确率值
plt.subplot(1, 2, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy Value')
plt.ylim([0, 1])
plt.legend()
# 绘制训练 & 验证的损失值
plt.subplot(1, 2, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("loss")
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()

plt.show()
