# Credit Card Fraud Detection using Conditional GANs

## Introduction

This project focuses on detecting credit card fraud using a Conditional Generative Adversarial Network (CGAN). CGANs are a type of generative model capable of generating synthetic data while conditioning on specific attributes. In this case, we condition the generation of synthetic credit card transactions on their class (fraudulent or non-fraudulent).

## Data Preprocessing

The data preprocessing steps include:

- Loading the credit card transaction dataset.
- Scaling the numerical features using StandardScaler and RobustScaler.
- Resampling the data to address class imbalance using SMOTE.
  
## Hyperparameter Optimization

To determine the best hyperparameters for training the CGAN, we use Optuna, an automatic hyperparameter optimization framework.

## Training the Conditional GAN

Training the CGAN involves:

- Defining the conditional generator and discriminator models.
- Compiling the discriminator with the Wasserstein loss.
- Implementing the training loop, including the Wasserstein loss and gradient penalty.
- Saving the best generator model based on the Frechet Inception Distance (FID) score.

## Generating Synthetic Data

After training the CGAN, you can use it to generate synthetic credit card transactions. The generated data can be used for various purposes, such as evaluating fraud detection models or augmenting the dataset.