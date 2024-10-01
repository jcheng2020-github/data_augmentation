# Author: Junfu Cheng
#Data augmentation typically involves techniques to increase the diversity of your dataset without actually collecting new data. Common methods include adding noise, scaling, and altering some features. Here's how you might apply some basic data augmentation techniques to your DataFrame:

#Adding Noise: Randomly perturbing numeric features.
#Scaling: Slightly increasing or decreasing numeric features by a small percentage.
#Random Sampling: Creating synthetic samples based on existing data.
#Key Notes:
#Noise Level: Adjust the noise level based on how much variance you want to introduce.
#Scaling Factor: Similarly, adjust the scaling factor as needed.
#Synthetic Sampling: You can also consider more advanced techniques like SMOTE (Synthetic Minority Over-sampling Technique) if your data is imbalanced.
#Make sure to validate the augmented data for consistency and relevance to your analysis or model training objectives!

import pandas as pd
import numpy as np

def augment_data(data, noise_level=0.01, num_augmented_rows=0, numerical_features=None):
    # Load the data into a DataFrame
    df = pd.DataFrame(data)
    
    # Define a function to add noise to numeric columns
    def add_noise(column, noise_level):
        noise = np.random.normal(0, noise_level * np.std(column), size=len(column))
        return column + noise

    # Create a copy of the original DataFrame for augmentation
    augmented_df = df.copy()

    # Determine numeric columns to augment
    if numerical_features is None:
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Augment specified numeric columns
    for col in numerical_features:
        if col in df.columns:
            augmented_df[col] = add_noise(augmented_df[col], noise_level)

    # Randomly sample existing rows without altering the original DataFrame
    sampled_rows = df.sample(n=num_augmented_rows, replace=True)  # Sample with replacement
    augmented_df = pd.concat([augmented_df, sampled_rows], ignore_index=True)

    # Combine original and augmented DataFrames
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    return combined_df
