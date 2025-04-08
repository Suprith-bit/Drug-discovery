import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
from torch.utils.data import Dataset, DataLoader

class SMILESDataset(Dataset):
    """Dataset class for handling SMILES strings data."""

    def __init__(self, csv_file, smiles_col='SMILES', property_cols=None, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with SMILES data.
            smiles_col (str): Name of column containing SMILES strings.
            property_cols (list): List of property column names to include.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.smiles_col = smiles_col
        self.property_cols = property_cols
        self.transform = transform

        # Filter out invalid SMILES
        valid_idx = []
        for i, smiles in enumerate(self.data_frame[smiles_col]):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_idx.append(i)

        self.data_frame = self.data_frame.iloc[valid_idx].reset_index(drop=True)

        # If property columns not specified, use all numeric columns except SMILES
        if self.property_cols is None:
            self.property_cols = [col for col in self.data_frame.columns
                                  if col != smiles_col and
                                  np.issubdtype(self.data_frame[col].dtype, np.number)]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        smiles = self.data_frame.iloc[idx][self.smiles_col]

        sample = {'smiles': smiles}

        if self.property_cols:
            properties = self.data_frame.iloc[idx][self.property_cols].values.astype(np.float32)
            sample['properties'] = properties

        if self.transform:
            sample = self.transform(sample)

        return sample

def get_dataloader(csv_file, batch_size=64, smiles_col='SMILES', property_cols=None,
                   shuffle=True, transform=None, num_workers=4):
    """
    Creates a DataLoader for SMILES data.

    Args:
        csv_file (str): Path to the CSV file with SMILES data.
        batch_size (int): Size of each batch.
        smiles_col (str): Name of column containing SMILES strings.
        property_cols (list): List of property column names to include.
        shuffle (bool): Whether to shuffle the data.
        transform (callable, optional): Optional transform to be applied on samples.
        num_workers (int): Number of worker threads for loading data.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    dataset = SMILESDataset(csv_file, smiles_col, property_cols, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

def load_zinc_dataset(file_path, subset_size=None):
    """
    Load ZINC dataset from file.

    Args:
        file_path (str): Path to the ZINC dataset file.
        subset_size (int, optional): Number of molecules to use (for testing).

    Returns:
        pandas.DataFrame: DataFrame containing SMILES and properties.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.smi'):
        df = pd.read_csv(file_path, delimiter=' ', header=None, names=['SMILES', 'ID'])
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Extract subset if specified
    if subset_size is not None and subset_size < len(df):
        df = df.sample(subset_size, random_state=42)

    # Validate SMILES
    valid_mask = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df[valid_mask].reset_index(drop=True)

    return df

def calculate_basic_properties(df, smiles_col='SMILES'):
    """
    Calculate basic molecular properties for all SMILES in dataframe.

    Args:
        df (pandas.DataFrame): DataFrame containing SMILES.
        smiles_col (str): Name of column containing SMILES strings.

    Returns:
        pandas.DataFrame: DataFrame with added property columns.
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    properties = []

    for smiles in result_df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # If invalid SMILES, add NaN values
            properties.append({
                'MolWt': np.nan,
                'LogP': np.nan,
                'NumHDonors': np.nan,
                'NumHAcceptors': np.nan,
                'NumRotatableBonds': np.nan
            })
        else:
            properties.append({
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
            })

    # Add properties to the dataframe
    properties_df = pd.DataFrame(properties)
    result_df = pd.concat([result_df, properties_df], axis=1)

    return result_df


def load_dataset(file_path, split_ratio=0.8, subset_size=None):
    """
    Load the dataset and split into train/test sets.

    Args:
        file_path: Path to dataset CSV file
        split_ratio: Train/test split ratio
        subset_size: Optional size limit for testing

    Returns:
        train_data, test_data: DataFrames for training and testing
    """
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.smi'):
        df = pd.read_csv(file_path, delimiter=' ', header=None, names=['SMILES', 'ID'])
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Take a subset if specified (useful for testing)
    if subset_size and subset_size < len(df):
        df = df.sample(subset_size, random_state=42)

    # Validate SMILES
    valid_mask = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df[valid_mask].reset_index(drop=True)

    # Calculate basic properties
    df = calculate_basic_properties(df)

    # Split into train/test
    train_size = int(len(df) * split_ratio)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    return train_data, test_data