import tensorflow as tf
import os
import glob
import numpy as np

# from tensorflow_examples.networks.pix2pix import pix2pix
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

# from IPython.display import clear_output
import matplotlib.pyplot as plt

# path = '../../DATA/oxford_iiit_pet/annotations/trimaps/'
# img_path = '../../DATA/oxford_iiit_pet/annotations/trimaps/Abyssinian_27.png'
# print(os.listdir(path)[-5:])
# img = tf.io.read_file(img_path)
# img = tf.image.decode_png(img)
# img = tf.squeeze(img)
# plt.subplot(121)
# plt.imshow(img)
#
# img1_path = '../../DATA/oxford_iiit_pet/images/Abyssinian_27.jpg'
# img1 = tf.io.read_file(img1_path)
# img1 = tf.image.decode_png(img1)
# plt.subplot(122)
# plt.imshow(img1)
# plt.show()

x_train_path = '..\\..\\DATA\\ISIC2016\\ISBI2016_ISIC_Part1_Training_Data\\*jpg'
y_train_path = '..\\..\\DATA\\ISIC2016\\ISBI2016_ISIC_Part1_Training_GroundTruth\\*.png'

# x_train_path = "..\\..\\DATA\\Oxford-IIIT Pet\\images\\*jpg"
# y_train_path = "..\\..\\DATA\\Oxford-IIIT Pet\\annotations\\trimaps\\*.png"

images = glob.glob(x_train_path)
anno = glob.glob(y_train_path)

images.sort(key=lambda x: x.split("/")[-1].split(".jpg")[0])
anno.sort(key=lambda x: x.split("/")[-1].split(".png")[0])
# 打乱
np.random.seed(2021)
index = np.random.permutation(len(images))
images = np.array(images)[index]
anno = np.array(anno)[index]


# 构建图片载入方法，主要包括读取原图像(jpg格式)，分割图像(png格式)，归一化函数和图像载入四个函数
def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img


# 归一化函数
def normal_img(input_images, input_anno):
    input_images = tf.cast(input_images, tf.float32) / 255.
    # input_anno = input_anno
    return input_images, input_anno


def load_image(input_images_path, input_anno_path):
    input_images = read_jpg(input_images_path)
    input_anno = read_png(input_anno_path)
    input_images = tf.image.resize(input_images, (224, 224))
    input_anno = tf.image.resize(input_anno, (224, 224))
    return normal_img(input_images, input_anno)


# 构建训练集和测试集，训练集的大小占总数据集的80%,bachsize=8，训练集有样本5912个，测试集有样本1478个。

# Datasets = tf.data.Datasets.from_tensor_slices((images, anno))
dataset = tf.data.Dataset.from_tensor_slices((images, anno))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# %%设置训练数据和验证集数据的大小
test_count = int(len(images) * 0.2)
train_count = len(images) - test_count
print(test_count, train_count)
# 跳过test_count个
train_dataset = dataset.skip(test_count)
test_dataset = dataset.take(test_count)

batch_size = 8
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据被充分打乱。
train_ds = train_dataset.shuffle(buffer_size=train_count).repeat().batch(batch_size)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_dataset.batch(batch_size)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print(train_ds)

for image, anno in train_ds.take(1):
    plt.subplot(1, 2, 1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))
    plt.subplot(1, 2, 2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(anno[0]))

plt.show()

vgg16 = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                    include_top=False,
                                    weights='imagenet')
vgg16.summary()


class Connect(tf.keras.layers.Layer):
    def __init__(self, filters=256, name='Connect', **kwargs):
        super(Connect, self).__init__(name=name, **kwargs)
        self.Conv_Transpose = tf.keras.layers.Convolution2DTranspose(filters=filters, kernel_size=3, strides=2,
                                                                     padding="same", activation="relu")
        self.conv_out = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")

    def call(self, inputs, **kwargs):
        x = self.Conv_Transpose(inputs)
        return self.conv_out(x)


layer_names = ["block5_conv3",
               "block4_conv3",
               "block3_conv3",
               "block5_pool"]
# 得到4个输出
layers_out = [vgg16.get_layer(layer_name).output for layer_name in layer_names]
multi_out_model = tf.keras.models.Model(inputs=vgg16.input,
                                        outputs=layers_out)
multi_out_model.trainable = False

# 创建输入
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
out_block5_conv3, out_block4_conv3, out_block3_conv3, out = multi_out_model(inputs)
print(out_block5_conv3.shape)

x1 = Connect(512, name="connect_1")(out)
x1 = tf.add(x1, out_block5_conv3)  # 元素对应相加

x2 = Connect(512, name="connect_2")(x1)
x2 = tf.add(x2, out_block4_conv3)  # 元素对应相加

x3 = Connect(256, name="connect_3")(x2)
x3 = tf.add(x3, out_block3_conv3)  # 元素对应相加

x4 = Connect(128, name="connect_4")(x3)

prediction = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding="same",
                                             activation="softmax")(x4)

model = tf.keras.models.Model(inputs=inputs, outputs=prediction)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["acc"])

steps_per_epoch = train_count // batch_size
validation_steps = test_count // batch_size

history = model.fit(train_ds, epochs=3,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_ds,
                    validation_steps=validation_steps)

for image, mask in test_ds.take(1):
    pred_mask = model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))
    plt.subplot(1, 3, 2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[0]))
    plt.subplot(1, 3, 3)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[0]))

plt.show()
