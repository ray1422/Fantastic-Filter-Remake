from dataset import get_dataset
from models import get_generator, get_discriminator, GANModel

BATCH_SIZE = 16
EPOCHS = 10
if __name__ == '__main__':
    gan = GANModel(generator=get_generator(), discriminator=get_discriminator())
    train_dataset, train_steps = get_dataset("dataset/train", batch_size=BATCH_SIZE)
    valid_dataset, valid_steps = get_dataset("dataset/valid", batch_size=BATCH_SIZE)
    gan.fit(train_dataset,
            steps_pre_epoch=train_steps,
            epochs=EPOCHS,
            valid_dataset=valid_dataset,
            valid_steps=valid_steps)
