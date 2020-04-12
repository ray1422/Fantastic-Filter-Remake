import glob
import os

import numpy
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataset(dir_path, batch_size=32):
    @tf.function
    def open_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        # image = tf.image.resize(image, [256, 256])
        image = tf.cast(image, dtype=tf.float32)
        image /= 127.5
        image -= 1  # normalize to [-1, 1] range
        return image

    def preprocess(x):
        x = tf.map_fn(open_image, x)
        return x

    filenames = [os.path.basename(f) for f in glob.glob(f"{dir_path}/x/*g")]  # jpg, jpeg or png LOL
    x_dir = f"{dir_path}/x"
    y_dir = f"{dir_path}/y"
    x_files = [f'{x_dir}/{i}' for i in filenames]
    y_files = [f'{y_dir}/{i}' for i in filenames]
    x_dataset = tf.data.Dataset.from_tensor_slices(x_files)
    y_dataset = tf.data.Dataset.from_tensor_slices(y_files)
    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
    dataset = dataset.cache().shuffle(buffer_size=len(filenames)).repeat().batch(batch_size=batch_size) \
        .map(preprocess).prefetch(AUTOTUNE)

    return dataset, len(filenames)
