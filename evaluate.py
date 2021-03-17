# coding = UTF-8

import os
import data
import glob
from models.losses import *

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K

# 可视化
import show

# load datasets
test_images_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Test_Data/*jpg'
test_masks_file_path = '../Datasets/ISIC2016/ISBI2016_ISIC_Part1_Test_GroundTruth/*.png'

images_path = glob.glob(test_images_file_path)
masks_path = glob.glob(test_masks_file_path)

# /////////// 测试用,请注释 ///////////
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用CPU计算

print('测试集个数:', len(images_path))  # 测试集个数: 379

# //////////////////////////////////////////////////////////
# 加载模型, 这样是为了在多个模型的情况下便于修改
# model_name = 'Unet'
# model_name = 'AttUnet'
# model_name = 'BCDUnet'
# model_name = 'ADUnet_L5'
model_name = 'ADUnet_L4'
# //////////////////////////////////////////////////////////
BATCH_SIZE = 2
dataset = tf.data.Dataset.from_tensor_slices((images_path, masks_path))
dataset = dataset.map(data.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = dataset.batch(BATCH_SIZE)
# x_train = list(map(lambda x: x[1], dataset))
# y_train = list(map(lambda x: x[1], dataset))

# -----------------------------------------------------------
K.set_learning_phase(0)
print(K.learning_phase())

# save path
save_model_path = 'output/' + model_name + '.h5'

model = models.load_model(save_model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef,
                                                           'f1_scores': f1_scores, 'precision_m': precision_m,
                                                           'recall_m': recall_m})

result = model.evaluate(test_dataset)

# 不同图片展示
show.show_predictions(save_model_path, test_dataset, num=10)
