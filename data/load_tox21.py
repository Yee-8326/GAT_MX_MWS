# !/usr/bin/env python3
"""
Script to download Tox21 dataset from its original source.
"""

import os
import requests
import zipfile
import pandas as pd


def download_file(url, save_path):
    """Download a file from a URL.

    Parameters
    ----------
    url : str
        URL to download from.
    save_path : str
        Path to save the file.
    """
    print(f"Downloading {url} to {save_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Download completed: {save_path}")


def main():
    """Main function to download Tox21 dataset."""
    print("=== Tox21 Dataset Downloader ===")

    # Create a directory to save the data
    save_dir = './tox21_data'
    os.makedirs(save_dir, exist_ok=True)

    # URL for Tox21 dataset (from dgllife library source code)
    # Based on dgllife implementation, Tox21 dataset is downloaded from:
    tox21_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'

    # Download the dataset
    zip_path = os.path.join(save_dir, 'tox21.csv.gz')
    download_file(tox21_url, zip_path)

    # Extract and read the CSV file
    print("\nExtracting and reading the dataset...")

    # Read the gzipped CSV file directly
    df = pd.read_csv(zip_path)

    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    # Save the extracted CSV if needed
    csv_path = os.path.join(save_dir, 'tox21.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDataset saved to: {csv_path}")

    # Split the dataset into train, val, test sets
    print("\n=== Splitting Dataset ===")
    from sklearn.model_selection import train_test_split

    # Split into train (80%), val (10%), test (10%)
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42)  # 0.1111 * 0.9 = 0.1

    print(f"Train set size: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"Validation set size: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")

    # Save the split datasets
    train_df.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(save_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(save_dir, 'test.csv'), index=False)

    print(f"\nSplit datasets saved to:")
    print(f"  - Train: {os.path.join(save_dir, 'train.csv')}")
    print(f"  - Validation: {os.path.join(save_dir, 'val.csv')}")
    print(f"  - Test: {os.path.join(save_dir, 'test.csv')}")

    print("\n=== Download and Processing Complete ===")


if __name__ == "__main__":
    main()
