import optuna
from data_preprocessing import load_and_preprocess_data
from hyperparam_optimization import optimize_hyperparameters

if __name__ == "__main__":
    X_resampled, _ = load_and_preprocess_data()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_hyperparameters(trial, X_resampled), n_trials=50)
    print(study.best_params)