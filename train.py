from dataset import get_dataset
from models import get_generator, get_discriminator, GANModel

if __name__ == '__main__':
        gan = GANModel(generator=get_generator(), discriminator=get_discriminator())
        gan.compile()
        train_dataset, train_steps = get_dataset("dataset/train", batch_size=1)
        valid_dataset, valid_steps = get_dataset("dataset/train", batch_size=1)
        gan.fit(train_dataset,
                steps_pre_epoch=train_steps,
                epochs=6000 // train_steps,
                valid_dataset=valid_dataset,
                valid_steps=valid_steps)
