import tensorflow as tf
import glob
import src.data as data
import Program.show as show
import Program.model as MYMODEL
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
from IPython.display import clear_output
import matplotlib.pyplot as plt

# dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
images_file_path = '..\\..\\DATA\\ISIC2016\\ISBI2016_ISIC_Part1_Training_Data\\*jpg'
masks_file_path = '..\\..\\DATA\\ISIC2016\\ISBI2016_ISIC_Part1_Training_GroundTruth\\*.png'

# images_file_path = "..\\..\\DATA\\Oxford-IIIT Pet\\images\\*jpg"
# masks_file_path = "..\\..\\DATA\\Oxford-IIIT Pet\\annotations\\trimaps\\*.png"

images_path = glob.glob(images_file_path)
masks_path = glob.glob(masks_file_path)

train_count = int(len(images_path) * 0.8)  # 80%
test_count = len(images_path) - train_count  # 20%
print('数据集个数:', len(images_path),
      '训练集个数:', train_count, '测试集个数:', test_count)

batch_size = 64
buffer_size = 1000
train_dataset, test_dataset, train, test = data.distribute(images_path, masks_path, test_count, batch_size, buffer_size)
steps_per_epoch = train_count // batch_size  # 900*0.8/64=11
print('测试集图片个数:', train_count, '每步epoch:', steps_per_epoch)

# train:  <SkipDataset shapes: ((128, 128, 3), (128, 128, 1)), types: (tf.float32, tf.float32)>
# test:   <TakeDataset shapes: ((128, 128, 3), (128, 128, 1)), types: (tf.float32, tf.float32)>
# train_dataset:  <PrefetchDataset shapes: ((None, 128, 128, 3), (None, 128, 128, 1)), types: (tf.float32, tf.float32)>
# test_dataset:   <BatchDataset shapes: ((None, 128, 128, 3), (None, 128, 128, 1)), types: (tf.float32, tf.float32)>


for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
show.display([sample_image, sample_mask])

# 输出信道数量为 3 是因为每个像素有三种可能的标签。把这想象成一个多类别分类，每个像素都将被分到三个类别当中。
OUTPUT_CHANNELS = 3

# 编码器是一个预训练的 MobileNetV2 模型，它在 tf.keras.applications 中已被准备好并可以直接使用。
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
# base_model = tf.keras.applications.MobileNetV2()
# <tensorflow.python.keras.engine.functional.Functional object at 0x7f56aa646668>

# 使用这些层的激活设置
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建特征提取模型
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

# 解码器/升频取样器是简单的一系列升频取样模块，在 TensorFlow examples 中曾被实施过。
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # 在模型中降频取样
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样然后建立跳跃连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # 这是模型的最后一层
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 快速浏览一下最终的模型架构：
tf.keras.utils.plot_model(model, show_shapes=True)


# 预测值
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            show.display([image[0], mask[0], create_mask(pred_mask)])
    else:
        show.display([sample_image, sample_mask,
                      create_mask(model.predict(sample_image[tf.newaxis, ...]))])


show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = test_count // batch_size // VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

# show_predictions(test_dataset, 3)
