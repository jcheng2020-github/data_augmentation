# Author: Junfu Cheng
import pandas as pd
import numpy as np

def augment_data(data, noise_level=0.01, scaling_factor=0.05, num_augmented_rows=0, numerical_features=None):
    """
    Augment the dataset with noise, scaling, and random sampling.
    
    Parameters:
        data (pd.DataFrame or np.ndarray): Input data to augment.
        noise_level (float): Level of noise to add to numeric features.
        scaling_factor (float): Factor to scale numeric features.
        num_augmented_rows (int): Number of synthetic samples to generate.
        numerical_features (list): List of numeric columns to augment.
        
    Returns:
        pd.DataFrame: Combined original and augmented DataFrame.
    """
    
    # Load the data into a DataFrame
    df = pd.DataFrame(data)
    
    # Define a function to add noise to numeric columns
    def add_noise(column, noise_level):
        noise = np.random.normal(0, noise_level * np.std(column), size=len(column))
        return column + noise

    # Define a function to scale numeric columns
    def scale_column(column, scaling_factor):
        scaling = np.random.uniform(1 - scaling_factor, 1 + scaling_factor, size=len(column))
        return column * scaling

    # Create a copy of the original DataFrame for augmentation
    augmented_df = df.copy()

    # Determine numeric columns to augment
    if numerical_features is None:
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Augment specified numeric columns
    for col in numerical_features:
        if col in df.columns:
            augmented_df[col] = add_noise(augmented_df[col], noise_level)
            augmented_df[col] = scale_column(augmented_df[col], scaling_factor)

    # Validate the number of rows to augment
    if num_augmented_rows > 0:
        if num_augmented_rows > len(df):
            raise ValueError("num_augmented_rows cannot be greater than the number of original rows.")
        
        # Randomly sample existing rows without altering the original DataFrame
        sampled_rows = df.sample(n=num_augmented_rows, replace=True)  # Sample with replacement
        augmented_df = pd.concat([augmented_df, sampled_rows], ignore_index=True)

    # Combine original and augmented DataFrames
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    return combined_df
