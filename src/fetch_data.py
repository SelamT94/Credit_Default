

import pandas as pd
import os
from ucimlrepo import fetch_ucirepo


def fetch_credit_card_dataset(dataset_id=350):
    # Fetch dataset
    default_of_credit_card_clients = fetch_ucirepo(id=dataset_id)
    
    # Extract data (as pandas dataframes)
    X = default_of_credit_card_clients.data.features
    y = default_of_credit_card_clients.data.targets
    
    # Extract metadata and variable information
    metadata = default_of_credit_card_clients.metadata
    variables = default_of_credit_card_clients.variables
    
    return X, y, metadata, variables


def get_combined_dataframe(dataset_id=350):
    
    X, y, metadata, variables = fetch_credit_card_dataset(dataset_id)
    
    # Combine features and target into a single dataframe
    df = pd.concat([X, y], axis=1)
    return df

def save_dataset_to_csv(dataset_id=350, output_path="data/credit_dataset.csv", save=True):

    # Get combined dataframe
    df = get_combined_dataframe(dataset_id)
    
    # Save to CSV if requested
    if save:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\n Dataset saved to {output_path}")
        print(f"   Shape: {df.shape}")
    
    return df


def print_dataset_info(dataset_id=350):

    X, y, metadata, variables = fetch_credit_card_dataset(dataset_id)
    
    print("Dataset Metadata:")
    print("=" * 50)
    print(metadata)
    print("\n" + "=" * 50 + "\n")
    
    print("Variable Information:")
    print("=" * 50)
    print(variables)
    print("\n" + "=" * 50 + "\n")
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns: {X.columns.tolist()}")
    print(f"Target columns: {y.columns.tolist()}")


if __name__ == "__main__":
    # Example usage
    print("Fetching Default of Credit Card Clients dataset from UCI ML Repository...\n")
    
    # Print dataset information
    print_dataset_info()
    
    # Get combined dataframe and save to CSV
    df = save_dataset_to_csv(
        dataset_id=350,
        output_path="data/credit_dataset.csv",
        save=True
    )
    
    print(f"\nCombined dataframe shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

