from train.train_conditional_gan import train_conditional_gan
from utils.data_preprocessing import load_and_preprocess_data


def objective(trial):
    X_resampled, y_resampled = load_and_preprocess_data()

    best_params = {
        'latent_dim': trial.suggest_int('latent_dim', 10, 50),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'n_critic': trial.suggest_int('n_critic', 1, 10),
        'batch_size': trial.suggest_int('batch_size', 16, 128)
    }

    best_fid = train_conditional_gan(X_resampled, best_params)

    return -best_fid