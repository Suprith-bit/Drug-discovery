
# src/utils/smiles_utils.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import re

class SMILESUtils:
    """
    Utility class for SMILES string manipulation.
    """
    @staticmethod
    def canonicalize_smiles(smiles):
        """
        Convert SMILES to canonical form.

        Args:
            smiles (str): SMILES string

        Returns:
            str: Canonical SMILES or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return None
        except:
            return None

    @staticmethod
    def preprocess_smiles(smiles_list, canonicalize=True, min_length=5, max_length=100):
        """
        Preprocess a list of SMILES strings.

        Args:
            smiles_list (list): List of SMILES strings
            canonicalize (bool): Whether to canonicalize SMILES
            min_length (int): Minimum SMILES length to keep
            max_length (int): Maximum SMILES length to keep

        Returns:
            list: Processed SMILES strings
        """
        processed_list = []

        for smiles in smiles_list:
            # Skip empty strings
            if not smiles or not isinstance(smiles, str):
                continue

            # Canonicalize if requested
            if canonicalize:
                smiles = SMILESUtils.canonicalize_smiles(smiles)
                if smiles is None:
                    continue

            # Filter by length
            if min_length <= len(smiles) <= max_length:
                processed_list.append(smiles)

        return processed_list

    @staticmethod
    def get_vocab_from_smiles(smiles_list):
        """
        Extract vocabulary from SMILES strings.

        Args:
            smiles_list (list): List of SMILES strings

        Returns:
            set: Set of unique characters
        """
        vocab = set()
        for smiles in smiles_list:
            # Process special tokens like Cl, Br
            for token in re.findall(r"Cl|Br|[^A-Z]|[A-Z]", smiles):
                vocab.add(token)

        return vocab

    @staticmethod
    def create_char_dict(smiles_list):
        """
        Create character to index mapping from SMILES strings.

        Args:
            smiles_list (list): List of SMILES strings

        Returns:
            dict: Dictionary mapping characters to indices
        """
        vocab = SMILESUtils.get_vocab_from_smiles(smiles_list)

        # Add special tokens
        all_tokens = ['<PAD>', '<START>', '<END>'] + sorted(list(vocab))

        # Create mapping
        char_dict = {c: i for i, c in enumerate(all_tokens)}

        return char_dict

    @staticmethod
    def smiles_to_sequence(smiles, char_dict, max_length=100):
        """
        Convert SMILES string to integer sequence.

        Args:
            smiles (str): SMILES string
            char_dict (dict): Character to index mapping
            max_length (int): Maximum sequence length

        Returns:
            np.array: Integer sequence
        """
        # Add start and end tokens
        sequence = ['<START>']

        # Process special tokens like Cl, Br
        tokens = re.findall(r"Cl|Br|[^A-Z]|[A-Z]", smiles)
        sequence.extend(tokens)

        sequence.append('<END>')

        # Convert to indices
        try:
            indices = [char_dict.get(c, char_dict['<PAD>']) for c in sequence]
        except KeyError as e:
            print(f"Error tokenizing: {smiles}, unknown character: {e}")
            indices = [char_dict['<START>'], char_dict['<END>']]

        # Pad sequence
        if len(indices) < max_length:
            indices += [char_dict['<PAD>']] * (max_length - len(indices))
        else:
            indices = indices[:max_length]

        return np.array(indices)

    @staticmethod
    def sequence_to_smiles(sequence, idx_to_char):
        """
        Convert integer sequence back to SMILES string.

        Args:
            sequence (list or np.array): Integer sequence
            idx_to_char (dict): Index to character mapping

        Returns:
            str: SMILES string
        """
        # Convert indices to characters
        chars = []
        for idx in sequence:
            if isinstance(idx, np.ndarray):
                idx = idx.item()

            char = idx_to_char.get(idx)

            # Stop at end token or padding
            if char == '<END>' or char == '<PAD>':
                break

            # Skip start token
            if char == '<START>':
                continue

            chars.append(char)

        # Join characters to form SMILES
        smiles = ''.join(chars)

        return smiles

    @staticmethod
    def one_hot_encode(sequence, vocab_size):
        """
        One-hot encode a sequence.

        Args:
            sequence (np.array): Integer sequence
            vocab_size (int): Size of vocabulary

        Returns:
            np.array: One-hot encoded sequence
        """
        one_hot = np.zeros((len(sequence), vocab_size))
        one_hot[np.arange(len(sequence)), sequence] = 1
        return one_hot


def decode_smiles():
    def decode_smiles(sequence, idx_to_char):
        """Convert a sequence of indices to a SMILES string."""
    # Skip start token (if present)
    if sequence[0] == 1:  # assuming 1 is the start token
        sequence = sequence[1:]

    # Convert until end token or end of sequence
    smiles = []
    for idx in sequence:
        if idx == 2:  # end token
            break
        if idx == 0:  # pad token
            continue
        if idx in idx_to_char:
            smiles.append(idx_to_char[idx])

    smiles_str = ''.join(smiles)

    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, isomericSmiles=True)
    return None