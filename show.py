import matplotlib.pyplot as plt
import tensorflow as tf
from networks import MyModel


# 图片展示，未来可能会加自动保存、自动关闭
def display(display_list):
    plt.figure()

    title = ['Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# 展示预测图。问题（注释的部分）
def show_predictions(checkpoint_save_path, dataset, num=1):
    MyModel.load_weights(checkpoint_save_path)
    for image, mask in dataset.take(num):
        pred_mask = MyModel.predict(image)
        # pred_mask = tf.argmax(pred_mask, axis=-1)
        # pred_mask = pred_mask[..., tf.newaxis]
        display([image[0], mask[0], pred_mask[0]])


# 准确率和loss曲线
def acc_loss(acc, val_acc, loss, val_loss):
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Value')
    plt.ylim([0, 1])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()

    plt.show()
