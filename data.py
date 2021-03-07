# 数据集模块
# 包括载入数据集(自制加载模块)、数据集增强
import tensorflow as tf


def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.cast(tf.io.decode_jpeg(img, channels=3), dtype=tf.float32) / 255.
    return img


def read_png(path):
    img = tf.io.read_file(path)
    img = tf.cast(tf.io.decode_png(img, channels=1), dtype=tf.float32) / 255.
    return img


# def normalize(input_image, input_mask):
#     input_image = tf.cast(input_image, tf.float32) / 255.0
#     input_mask = tf.cast(input_mask, tf.float32) / 255.0
#     return input_image, input_mask


@tf.function
def load_image_train(input_images_path, input_masks_path):
    input_images = read_jpg(input_images_path)
    input_masks = read_png(input_masks_path)
    input_images = tf.image.resize(input_images, (256, 256))
    input_masks = tf.image.resize(input_masks, (256, 256))

    if tf.random.uniform(()) > 0.5:
        input_images = tf.image.flip_left_right(input_images)
        input_masks = tf.image.flip_left_right(input_masks)

    # input_images, input_masks = normalize(input_images, input_masks)

    return input_images, input_masks


def load_image_test(input_images_path, input_masks_path):
    input_images = read_jpg(input_images_path)
    input_masks = read_png(input_masks_path)
    input_images = tf.image.resize(input_images, (256, 256))
    input_masks = tf.image.resize(input_masks, (256, 256))

    # input_images, input_masks = normalize(input_images, input_masks)

    return input_images, input_masks
