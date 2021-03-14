# coding = UTF-8

import os
from models.networks import MyModel, Unet, BCDUnet
from models.losses import *
# 评价
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import f1_score

save_path = 'output/ADUNet_Qian.h5'
print('-------------下载 ' + save_path + '-------------')
# 下载训练好的模型
model = models.load_model(save_path,
                          custom_objects={'dice_coef_loss': dice_coef_loss,
                                          'dice_coef': dice_coef})

# Read
NUM_PARTS = 2
parts = []


def read_map_fn(x):
    return tf.io.parse_tensor(x, tf.float32)


for i in range(NUM_PARTS):
    parts.append(tf.data.TFRecordDataset(f'output/mydata_{i}.tfrecord').map(read_map_fn))
test_dataset = tf.data.Dataset.zip(tuple(parts))
print(test_dataset)


scores = model.evaluate(test_dataset, steps=1, verbose=1)
print('Test loss:\t', scores[0])
print('Test binary_accuracy:\t', scores[1])
print('Test dice_coef:\t', scores[2])

predictions = model.predict(test_dataset, batch_size=8, verbose=1)

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(f1_score(y_true, y_pred, average='macro'))
