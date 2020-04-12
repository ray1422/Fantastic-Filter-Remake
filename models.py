import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from numpy import inf
from tensorflow.keras.utils import Progbar
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
import tensorflow as tf
import tensorflow.keras.backend as K


def get_generator():
    def _res_block(_x, filters):
        _x1 = Conv2D(filters=filters, kernel_size=3, activation=None, padding='SAME', use_bias=False)(_x)
        _x1 = Conv2D(filters=filters, kernel_size=3, activation=None, padding='SAME', use_bias=False)(_x1)
        _x1 = tf.nn.leaky_relu(BatchNormalization()(_x1))
        return _x + _x1

    inputs = Input(shape=(None, None, 3), dtype=tf.float32)
    x = inputs
    x = Conv2D(kernel_size=5, filters=64, padding='SAME', activation=None, use_bias=False)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))
    x = Conv2D(kernel_size=3, filters=128, strides=2, padding='SAME', activation=None, use_bias=False)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))
    x = Conv2D(kernel_size=3, filters=256, strides=2, padding='SAME', activation=None, use_bias=False)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))

    for i in range(9):
        x = _res_block(x, 256)
    x = tf.image.resize(x, K.shape(x)[1:3] * 2)
    x = Conv2D(kernel_size=3, filters=128, strides=1, padding='SAME', activation=None, use_bias=False)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))
    x = tf.image.resize(x, K.shape(x)[1:3] * 2)
    x = Conv2D(kernel_size=3, filters=64, strides=1, padding='SAME', activation=None, use_bias=False)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))
    x = Conv2D(kernel_size=3, filters=3, strides=1, padding='SAME', activation=tf.nn.tanh)(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def get_discriminator():
    inputs = Input(shape=(None, None, 3))
    x = inputs
    x = Conv2D(kernel_size=4, filters=64, strides=2, padding="SAME", use_bias=False, activation=None)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))
    x = Conv2D(kernel_size=4, filters=128, strides=2, padding="SAME", use_bias=False, activation=None)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))
    x = Conv2D(kernel_size=4, filters=256, strides=2, padding="SAME", use_bias=False, activation=None)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))
    x = Conv2D(kernel_size=4, filters=512, strides=2, padding="SAME", use_bias=False, activation=None)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))
    x = Conv2D(kernel_size=4, filters=512, strides=2, padding="SAME", use_bias=False, activation=None)(x)
    x = tf.nn.leaky_relu(BatchNormalization()(x))
    x = Conv2D(kernel_size=4, filters=1, padding="SAME", activation=None)(x)

    model = Model(inputs=inputs, outputs=x)
    return model


class GANModel:
    def __init__(self, generator: Model, discriminator: Model):
        self.generator = generator
        self.discriminator = discriminator

    def compile(self):
        optimizer = Adam(learning_rate=1e-4)
        self.generator.compile(loss=lambda *args: GANModel.generator_loss(self, *args), optimizer=optimizer)
        self.discriminator.compile(loss=lambda *args: GANModel.discriminator_loss(self, *args), optimizer=optimizer)

    def generator_loss(self, y_true, y_pred, *args, **kwargs):
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        gan_loss = tf.square(1 - self.discriminator(y_pred))
        loss = l1_loss + gan_loss
        return loss

    def discriminator_loss(self, y_true, y_pred, *args, **kwargs):
        # y_true: X
        # y_pred: predict of Y
        fake_pred = self.discriminator(self.generator(y_true))
        real_loss = tf.reduce_mean(tf.square(tf.ones_like(tf.shape(y_pred), dtype=tf.float32) - y_pred))
        fake_loss = tf.reduce_mean(tf.square(fake_pred))
        loss = (real_loss + fake_loss) / 2
        return loss

    def fit(self, train_dataset, valid_dataset: [tf.data.Dataset, None] = None, steps_pre_epoch=-1,
            valid_steps=-1, epochs=1,
            valids_pre_steps=-1,
            save_best=True, checkpoints_dir="./checkpoints"):
        print(f"train for {steps_pre_epoch} steps, valid for {valid_steps} steps.")
        if steps_pre_epoch == -1:
            raise ValueError("step_pre_epoch is required.")
        if valid_dataset is not None and valid_steps == -1:
            raise ValueError("valid_steps is required.")
        valid_loss = inf
        td_it = iter(train_dataset)

        for epoch in range(epochs):
            print(f"epoch {epoch}/{epochs}")
            pb = Progbar(target=steps_pre_epoch, stateful_metrics=['G_loss', 'D_loss', 'valid_G_loss', 'valid_D_loss'])
            for local_step in range(steps_pre_epoch):
                x, y = next(td_it)
                g_loss = self.generator.train_on_batch(x=x, y=y)
                d_loss = self.discriminator.train_on_batch(x=y, y=x)
                pb.update(local_step, values=[('G_loss', g_loss), ('D_loss', d_loss)])
                if (local_step == steps_pre_epoch - 1 or (valids_pre_steps > 0 and local_step % valids_pre_steps == 0)) \
                                                    and valid_dataset is not None:
                    valid_dataset: tf.data.Dataset
                    valid_g_loss = self.generator.evaluate(valid_dataset, verbose=0, steps=valid_steps)
                    valid_d_loss = self.discriminator.evaluate(valid_dataset.map(lambda _x, _y: (_y, _x)), verbose=0,
                                                               steps=valid_steps)
                    pb.update(local_step + 1, [('valid_G_loss', valid_g_loss), ('valid_D_loss', valid_d_loss)])
