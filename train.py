from dataset import get_dataset
from models import get_generator, get_discriminator, GANModel

if __name__ == '__main__':
    gan = GANModel(generator=get_generator(), discriminator=get_discriminator())
    gan.compile()
    train_dataset, train_steps = get_dataset("dataset/train", batch_size=8)
    valid_dataset, valid_steps = get_dataset("dataset/valid", batch_size=8)
    gan.fit(train_dataset,
            steps_pre_epoch=train_steps,
            epochs=5,
            valid_dataset=valid_dataset,
            valid_steps=valid_steps)
