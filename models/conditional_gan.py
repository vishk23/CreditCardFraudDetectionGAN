from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, LeakyReLU
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from conditional_generator import build_conditional_generator
from conditional_discriminator import build_conditional_discriminator

from utils.loss_calculator import wasserstein_loss

def build_conditional_gan(latent_dim, learning_rate):
    # Input layers
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')

    # Generator
    generator = build_conditional_generator(latent_dim)
    generated_data = generator([noise, label])

    # Discriminator
    discriminator = build_conditional_discriminator()
    discriminator.trainable = False  # Freeze discriminator during GAN training
    validity = discriminator([generated_data, label])

    # GAN model
    conditional_gan = Model([noise, label], validity)
    conditional_gan.compile(optimizer=Adam(learning_rate, 0.5), loss=wasserstein_loss)

    return conditional_gan
