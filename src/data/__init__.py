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
        self.data_frame = pd.read_csv(csv_file)
        self.smiles_col = smiles_col
        self.property_cols = property_cols
        self.transform = transform

        valid_idx = [i for i, smiles in enumerate(self.data_frame[smiles_col]) if Chem.MolFromSmiles(smiles) is not None]
        self.data_frame = self.data_frame.iloc[valid_idx].reset_index(drop=True)

        if self.property_cols is None:
            self.property_cols = [col for col in self.data_frame.columns if col != smiles_col and np.issubdtype(self.data_frame[col].dtype, np.number)]

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

def get_dataloader(csv_file, batch_size=64, smiles_col='SMILES', property_cols=None, shuffle=True, transform=None, num_workers=4):
    dataset = SMILESDataset(csv_file, smiles_col, property_cols, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_zinc_dataset(file_path, subset_size=None):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.smi'):
        df = pd.read_csv(file_path, delimiter=' ', header=None, names=['SMILES', 'ID'])
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    if subset_size is not None and subset_size < len(df):
        df = df.sample(subset_size, random_state=42)

    valid_mask = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    return df[valid_mask].reset_index(drop=True)

def calculate_basic_properties(df, smiles_col='SMILES'):
    result_df = df.copy()
    properties = []

    for smiles in result_df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            properties.append({'MolWt': np.nan, 'LogP': np.nan, 'NumHDonors': np.nan, 'NumHAcceptors': np.nan, 'NumRotatableBonds': np.nan})
        else:
            properties.append({'MolWt': Descriptors.MolWt(mol), 'LogP': Descriptors.MolLogP(mol), 'NumHDonors': Descriptors.NumHDonors(mol), 'NumHAcceptors': Descriptors.NumHAcceptors(mol), 'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)})

    return pd.concat([result_df, pd.DataFrame(properties)], axis=1)

class SMILESPreprocessor:
    def __init__(self, max_length=100, padding=True, canonical=True):
        self.max_length = max_length
        self.padding = padding
        self.canonical = canonical
        self.char_dict = None
        self.vocab_size = 0

    def fit(self, smiles_list):
        if self.canonical:
            smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True) for s in smiles_list if Chem.MolFromSmiles(s)]

        char_set = {char for smiles in smiles_list for char in smiles}
        self.char_dict = {'<pad>': 0, '<start>': 1, '<end>': 2, **{char: i + 3 for i, char in enumerate(sorted(char_set))}}
        self.vocab_size = len(self.char_dict)
        self.reverse_char_dict = {v: k for k, v in self.char_dict.items()}
        return self

    def transform(self, smiles):
        if self.canonical:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        sequence = [self.char_dict.get('<start>')] + [self.char_dict.get(char, 0) for char in smiles[:self.max_length-2]] + [self.char_dict.get('<end>')]
        if self.padding:
            sequence += [self.char_dict.get('<pad>')] * (self.max_length - len(sequence))
        return np.array(sequence)

    def transform_batch(self, smiles_list):
        return np.array([self.transform(s) for s in smiles_list])

    def inverse_transform(self, sequence):
        sequence = sequence.tolist()
        sequence = sequence[:sequence.index(0)] if 0 in sequence else sequence
        sequence = sequence[:sequence.index(2)] if 2 in sequence else sequence
        return ''.join([self.reverse_char_dict.get(idx, '') for idx in sequence[1:]])

def augment_smiles(smiles, n_augmentations=5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    return list(set([Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)), doRandom=True, isomericSmiles=True) for _ in range(n_augmentations)])) + [smiles]

# __init__.py
__all__ = ['SMILESDataset', 'get_dataloader', 'load_zinc_dataset', 'calculate_basic_properties', 'SMILESPreprocessor', 'augment_smiles']
