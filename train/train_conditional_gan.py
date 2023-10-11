import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from data_preprocessing import load_and_preprocess_data

from models.conditional_discriminator import build_conditional_discriminator
from models.conditional_generator import build_conditional_generator
from models.conditional_gan import build_conditional_gan

from utils.fid_calculator import compute_fid  
from utils.loss_calculator import wasserstein_loss  

def train_conditional_gan(X_resampled, best_params):
    conditional_generator = build_conditional_generator(best_params['latent_dim'])
    conditional_discriminator = build_conditional_discriminator()
    conditional_discriminator.compile(optimizer=Adam(best_params['learning_rate'], 0.5), loss=wasserstein_loss)

    noise = Input(shape=(best_params['latent_dim'],))
    label = Input(shape=(1,), dtype='int32')
    
    conditional_gan = build_conditional_gan(best_params['latent_dim'], best_params['learning_rate'])

    epochs = 100
    batch_size = best_params['batch_size']
    n_critic = best_params['n_critic']
    
    best_fid = float('inf')  # Initialize best_fid

    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = np.random.randint(0, X_resampled.shape[0], batch_size)
            real_samples = X_resampled.iloc[idx]
            real_labels = np.ones((batch_size, 1))

            noise = np.random.normal(0, 1, (batch_size, best_params['latent_dim']))
            synthetic_labels = np.random.randint(0, 2, batch_size)
            synthetic_samples = conditional_generator.predict([noise, synthetic_labels])
            fake_labels = -np.ones((batch_size, 1))

            d_loss_real = conditional_discriminator.train_on_batch([real_samples, real_labels], real_labels)
            d_loss_fake = conditional_discriminator.train_on_batch([synthetic_samples, synthetic_labels], fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, best_params['latent_dim']))
        valid_y = np.ones((batch_size, 1))
        synthetic_labels = np.random.randint(0, 2, batch_size)
        g_loss = conditional_gan.train_on_batch([noise, synthetic_labels], valid_y)

        noise = np.random.normal(0, 1, (1000, best_params['latent_dim']))
        synthetic_labels = np.random.randint(0, 2, 1000)
        synthetic_samples = conditional_generator.predict([noise, synthetic_labels])
        real_samples = X_resampled.sample(n=1000)
        real_samples_tensor = tf.convert_to_tensor(real_samples.values, dtype=tf.float32)
        current_fid = compute_fid(real_samples_tensor, synthetic_samples)

        if current_fid < best_fid:
            best_fid = current_fid
            conditional_generator.save('best_generator.h5')

    num_samples = 1000
    noise = np.random.normal(0, 1, (num_samples, best_params['latent_dim']))
    synthetic_labels = np.random.randint(0, 2, num_samples)
    synthetic_samples = conditional_generator.predict([noise, synthetic_labels])

    synthetic_df = pd.DataFrame(synthetic_samples, columns=X_resampled.columns)

    print("Statistics of Original Data:")
    print(X_resampled.describe())
    print("\nStatistics of Synthetic Data:")
    print(synthetic_df.describe())

    return best_fid