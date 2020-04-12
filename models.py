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
        _x1 = Conv2D(filters=filters, kernel_size=3, activation=None, padding='SAME', )(_x)
        _x1 = Conv2D(filters=filters, kernel_size=3, activation=None, padding='SAME', )(_x1)
        _x1 = tf.nn.leaky_relu(_x1)
        return _x + _x1

    inputs = Input(shape=(None, None, 3), dtype=tf.float32)
    x = inputs
    x = Conv2D(kernel_size=5, filters=64, padding='SAME', activation=None, )(x)
    x = tf.nn.leaky_relu(x)
    x = Conv2D(kernel_size=3, filters=128, strides=2, padding='SAME', activation=None, )(x)
    x = tf.nn.leaky_relu(x)
    x = Conv2D(kernel_size=3, filters=256, strides=2, padding='SAME', activation=None, )(x)
    x = tf.nn.leaky_relu(x)

    for i in range(9):
        x = _res_block(x, 256)
    x = tf.image.resize(x, K.shape(x)[1:3] * 2)
    x = Conv2D(kernel_size=3, filters=128, strides=1, padding='SAME', activation=None, )(x)
    x = tf.nn.leaky_relu(x)
    x = tf.image.resize(x, K.shape(x)[1:3] * 2)
    x = Conv2D(kernel_size=3, filters=64, strides=1, padding='SAME', activation=None, )(x)
    x = tf.nn.leaky_relu(x)
    x = Conv2D(kernel_size=3, filters=3, strides=1, padding='SAME', activation=tf.nn.tanh)(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def get_discriminator():
    inputs = Input(shape=(None, None, 3))
    x = inputs
    x = Conv2D(kernel_size=4, filters=64, strides=2, padding="SAME", activation=None)(x)
    x = tf.nn.leaky_relu((x))
    x = Conv2D(kernel_size=4, filters=128, strides=2, padding="SAME", activation=None)(x)
    x = tf.nn.leaky_relu((x))
    x = Conv2D(kernel_size=4, filters=256, strides=2, padding="SAME", activation=None)(x)
    x = tf.nn.leaky_relu((x))
    x = Conv2D(kernel_size=4, filters=512, strides=2, padding="SAME", activation=None)(x)
    x = tf.nn.leaky_relu((x))
    x = Conv2D(kernel_size=4, filters=512, strides=2, padding="SAME", activation=None)(x)
    x = tf.nn.leaky_relu((x))
    x = Conv2D(kernel_size=4, filters=1, padding="SAME", activation=None)(x)

    model = Model(inputs=inputs, outputs=x)
    return model


class GANModel:
    def __init__(self, generator: Model, discriminator: Model):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optm = Adam(learning_rate=1e-4)
        self.d_optm = Adam(learning_rate=1e-4)

    def train_generator(self, x, y):
        with tf.GradientTape() as tape:
            gen = self.generator(x, training=True)
            gen_pred = self.discriminator(gen)
            gan_loss = tf.reduce_mean(tf.square(1 - gen_pred))
            l2_loss = tf.reduce_mean(tf.abs(gen - y))
            loss = gan_loss + l2_loss
        grads = tape.gradient(loss, self.generator.trainable_weights)
        self.g_optm.apply_gradients(zip(grads, self.generator.trainable_weights))
        return loss, l2_loss, gen

    def train_discriminator(self, x, y):
        # where x is the input of G, y is the real image.
        with tf.GradientTape() as tape:
            fake_pred = self.discriminator(x)
            real_pred = self.discriminator(y)
            real_loss = tf.reduce_mean(tf.square(1 - real_pred))
            fake_loss = tf.reduce_mean(tf.square(fake_pred))
            loss = (fake_loss + real_loss) / 2
        grads = tape.gradient(loss, self.discriminator.trainable_weights)
        self.d_optm.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        return loss, fake_pred, real_pred

    def evaluate_generator(self, dataset, steps=-1):
        losses = []
        for i, (x, y) in enumerate(dataset):
            if i >= steps:
                break
            x = self.generator(x, training=False)
            loss = tf.reduce_mean(tf.abs(y - x))
            losses.append(loss)
        return tf.reduce_mean(losses)

    def evaluate_discriminator(self, dataset, steps=-1):
        losses = []
        for i, (x, y) in enumerate(dataset):
            if i >= steps:
                break
            x = self.generator(x, training=False)
            real_pred = self.discriminator(y)
            fake_pred = self.discriminator(x)
            real_loss = tf.reduce_mean(tf.square(1 - real_pred))
            fake_loss = tf.reduce_mean(tf.square(fake_pred))
            loss = (real_loss + fake_loss) / 2
            losses.append(loss)
        return tf.reduce_mean(losses)

    def fit(self, train_dataset, steps_pre_epoch=-1,
            valid_dataset: [tf.data.Dataset, None] = None, valid_steps=-1,
            epochs=1, valid_pre_steps=-1,
            log_dir="./logs",
            save_best=True, checkpoints_dir="./checkpoints"):
        print(f"train for {steps_pre_epoch} steps, valid for {valid_steps} steps.")
        if steps_pre_epoch == -1:
            raise ValueError("step_pre_epoch is required.")
        if valid_dataset is not None and valid_steps == -1:
            raise ValueError("valid_steps is required.")
        valid_loss_best = inf
        td_it = iter(train_dataset)
        train_writer = valid_writer = None
        if log_dir is not None:
            train_writer = tf.summary.create_file_writer(f"{log_dir}/train")
            valid_writer = tf.summary.create_file_writer(f"{log_dir}/val")
        for epoch in range(epochs):
            print(f"epoch {epoch} / {epochs}")
            pb = Progbar(target=steps_pre_epoch, stateful_metrics=['G_loss', 'D_loss', 'valid_G_loss', 'valid_D_loss'])
            for local_step in range(steps_pre_epoch):
                x, y = next(td_it)
                g_loss, g_l2_loss, gen = self.train_generator(x, y)
                d_loss, pred_fake, pred_real = self.train_discriminator(gen, y)
                if train_writer is not None:
                    if local_step % 100 == 0 or (local_step % 50 == 0 and local_step < 300):
                        step = local_step + steps_pre_epoch * epoch
                        with train_writer.as_default():
                            tf.summary.scalar("Generator/loss_l2", g_l2_loss.numpy(), step=step)
                            tf.summary.scalar("Generator/loss", g_loss.numpy(), step=step)
                            tf.summary.scalar("Discriminator/loss", d_loss.numpy(), step=step)
                            tf.summary.image("images/_input", x, step=step, max_outputs=1)
                            tf.summary.image("images/enhanced", gen, step=step, max_outputs=1)
                            tf.summary.image("images/target", y, step=step, max_outputs=1)
                            tf.summary.histogram("Discriminator/real", pred_real, step=step)
                            tf.summary.histogram("Discriminator/fake", pred_fake, step=step)

                pb.update(local_step, values=[('G_loss', g_loss), ('D_loss', d_loss)])
                if (local_step == steps_pre_epoch - 1 or (valid_pre_steps > 0 and local_step % valid_pre_steps == 0)) \
                        and valid_dataset is not None:
                    valid_dataset: tf.data.Dataset
                    valid_g_loss = self.evaluate_generator(valid_dataset, steps=valid_steps)
                    valid_d_loss = self.evaluate_discriminator(valid_dataset, steps=valid_steps)
                    if valid_writer is not None:
                        with valid_writer.as_default():
                            step = local_step + steps_pre_epoch * epoch
                            tf.summary.scalar("Generator/loss_l2", valid_g_loss.numpy(), step=step)
                            tf.summary.scalar("Discriminator/loss", d_loss.numpy(), step=step)

                    pb.update(local_step + 1, [('valid_G_loss', valid_g_loss), ('valid_D_loss', valid_d_loss)])

                    if not save_best:
                        self.generator.save(f'{checkpoints_dir}/generator.h5')
                        self.discriminator.save(f"{checkpoints_dir}.discriminator.h5")
                    else:
                        if valid_g_loss < valid_loss_best:
                            print(f"loss improved from {valid_loss_best} to {valid_g_loss}. model saved.")
                            valid_loss_best = valid_g_loss
                            self.generator.save(f'{checkpoints_dir}/generator.h5')
                            self.discriminator.save(f"{checkpoints_dir}.discriminator.h5")
