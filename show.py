import matplotlib.pyplot as plt
import tensorflow as tf
from models.losses import *
from tensorflow.keras import models


# 图片展示，未来可能会加自动保存、自动关闭
def display_sample(display_list):
    plt.figure()

    title = ['Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# 展示预测图。问题（注释的部分）
def show_predictions(path, dataset, num=1):
    model = models.load_model(path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef,
                                                    'f1_scores': f1_scores, 'precision_m': precision_m,
                                                    'recall_m': recall_m})
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        # pred_mask = tf.argmax(pred_mask, axis=-1)
        # pred_mask = pred_mask[..., tf.newaxis]
        display_sample([image[0], mask[0], pred_mask[0]])
