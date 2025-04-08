import numpy as np
from rdkit import Chem
import re
from collections import defaultdict

class SMILESPreprocessor:
    """Class for preprocessing SMILES strings for deep learning models."""

    def __init__(self, max_length=100, padding=True, canonical=True):
        """
        Initialize the preprocessor.

        Args:
            max_length (int): Maximum length of SMILES string after tokenization.
            padding (bool): Whether to pad sequences to max_length.
            canonical (bool): Whether to convert SMILES to canonical form.
        """
        self.max_length = max_length
        self.padding = padding
        self.canonical = canonical
        self.char_dict = None
        self.vocab_size = 0

    def fit(self, smiles_list):
        """
        Build vocabulary from list of SMILES strings.

        Args:
            smiles_list (list): List of SMILES strings.

        Returns:
            self: The preprocessor object.
        """
        # Canonicalize SMILES if needed
        if self.canonical:
            smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
                           for s in smiles_list if Chem.MolFromSmiles(s)]

        # Create vocabulary
        char_set = set()
        for smiles in smiles_list:
            for char in smiles:
                char_set.add(char)

        # Sort for reproducibility
        chars = sorted(list(char_set))

        # Create char-to-index mapping
        self.char_dict = {'<pad>': 0, '<start>': 1, '<end>': 2}
        for i, char in enumerate(chars):
            self.char_dict[char] = i + 3  # +3 because of special tokens

        self.vocab_size = len(self.char_dict)
        self.reverse_char_dict = {v: k for k, v in self.char_dict.items()}

        return self

    def transform(self, smiles):
        """
        Convert a SMILES string to a numeric sequence.

        Args:
            smiles (str): Input SMILES string.

        Returns:
            np.array: Numeric sequence.
        """
        if self.canonical:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        # Convert to numeric sequence
        sequence = [self.char_dict.get('<start>')]
        for char in smiles[:self.max_length-2]:  # -2 for start and end tokens
            if char in self.char_dict:
                sequence.append(self.char_dict.get(char))
            else:
                sequence.append(self.char_dict.get('?', 0))  # Unknown char

        sequence.append(self.char_dict.get('<end>'))

        # Pad sequence if needed
        if self.padding:
            sequence = sequence + [self.char_dict.get('<pad>')] * (self.max_length - len(sequence))

        return np.array(sequence)

    def transform_batch(self, smiles_list):
        """
        Convert a batch of SMILES strings to numeric sequences.

        Args:
            smiles_list (list): List of SMILES strings.

        Returns:
            np.array: Batch of numeric sequences.
        """
        return np.array([self.transform(s) for s in smiles_list])

    def inverse_transform(self, sequence):
        """
        Convert a numeric sequence back to a SMILES string.

        Args:
            sequence (np.array): Numeric sequence.

        Returns:
            str: SMILES string.
        """
        # Remove padding and end token
        sequence = sequence.tolist()
        if 0 in sequence:
            sequence = sequence[:sequence.index(0)]
        if 2 in sequence:  # <end> token
            sequence = sequence[:sequence.index(2)]

        # Remove start token if present
        if sequence and sequence[0] == 1:  # <start> token
            sequence = sequence[1:]

        # Convert back to SMILES
        smiles = ''.join([self.reverse_char_dict.get(idx, '') for idx in sequence])

        return smiles

    def filter_valid_smiles(self, generated_sequences):
        """
        Filter valid SMILES from generated sequences.

        Args:
            generated_sequences (np.array): Batch of generated sequences.

        Returns:
            list: List of valid SMILES strings.
        """
        valid_smiles = []

        for seq in generated_sequences:
            smiles = self.inverse_transform(seq)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                valid_smiles.append(canonical_smiles)

        return valid_smiles

def augment_smiles(smiles, n_augmentations=5):
    """
    Create augmented versions of a SMILES string by generating
    random SMILES for the same molecule.

    Args:
        smiles (str): Input SMILES string.
        n_augmentations (int): Number of augmentations to create.

    Returns:
        list: List of augmented SMILES strings.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]  # Return original if invalid

    augmented = []
    for _ in range(n_augmentations):
        # Set random atom ordering
        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if new_mol is None:
            continue

        # Generate a random SMILES with random atom ordering
        random_smiles = Chem.MolToSmiles(new_mol, doRandom=True, isomericSmiles=True)
        augmented.append(random_smiles)

    # Deduplicate and add original
    augmented = list(set(augmented))
    augmented.append(smiles)

    return augmented


def preprocess_smiles_dataset():
    return None