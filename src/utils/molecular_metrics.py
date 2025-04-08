# src/utils/molecular_metrics.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors
import pandas as pd
import pickle
import os
import torch

class MolecularMetrics:
    """
    Utility class for calculating various molecular properties and metrics.
    """
    def __init__(self):
        """Initialize the MolecularMetrics class."""
        pass

    @staticmethod
    def validate_molecule(smiles):
        """
        Check if a SMILES string represents a valid molecule.

        Args:
            smiles (str): SMILES string

        Returns:
            bool: True if valid, False otherwise
        """
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    @staticmethod
    def calculate_validity_score(smiles_list):
        """
        Calculate the validity ratio of a list of SMILES strings.

        Args:
            smiles_list (list): List of SMILES strings

        Returns:
            float: Ratio of valid molecules
        """
        valid_count = sum(1 for smiles in smiles_list if MolecularMetrics.validate_molecule(smiles))
        return valid_count / len(smiles_list) if len(smiles_list) > 0 else 0.0

    @staticmethod
    def calculate_drug_likeness(mol):
        """
        Calculate QED (Quantitative Estimate of Drug-likeness) for a molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object

        Returns:
            float: QED score between 0 and 1
        """
        if mol is None:
            return 0.0
        try:
            return QED.qed(mol)
        except:
            return 0.0

    @staticmethod
    def calculate_sa_score(mol):
        """
        Calculate synthetic accessibility score.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object

        Returns:
            float: SA score (lower is better)
        """
        from rdkit.Chem import rdMolDescriptors

        if mol is None:
            return 10.0  # Worst score for invalid molecules

        # Try to import SA Score
        try:
            from rdkit.Chem.Descriptors import sasecore
            return sasecore.calculateScore(mol)
        except:
            # Fallback to a simpler metric if SA Score is not available
            return 5.0

    @staticmethod
    def calculate_logp(mol):
        """
        Calculate octanol-water partition coefficient (LogP).

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object

        Returns:
            float: LogP value
        """
        if mol is None:
            return 0.0
        return Crippen.MolLogP(mol)

    @staticmethod
    def calculate_mw(mol):
        """
        Calculate molecular weight.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object

        Returns:
            float: Molecular weight
        """
        if mol is None:
            return 0.0
        return Descriptors.MolWt(mol)

    @staticmethod
    def calculate_num_rotatable_bonds(mol):
        """
        Calculate number of rotatable bonds.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object

        Returns:
            int: Number of rotatable bonds
        """
        if mol is None:
            return 0
        return Descriptors.NumRotatableBonds(mol)

    @staticmethod
    def check_lipinski_rule_of_five(mol):
        """
        Check if a molecule follows Lipinski's Rule of Five.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object

        Returns:
            dict: Dictionary with Rule of Five criteria
        """
        if mol is None:
            return {
                "MW <= 500": False,
                "LogP <= 5": False,
                "H-bond donors <= 5": False,
                "H-bond acceptors <= 10": False,
                "Passes Rule of 5": False
            }

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)

        passes_mw = mw <= 500
        passes_logp = logp <= 5
        passes_donors = h_donors <= 5
        passes_acceptors = h_acceptors <= 10

        # A molecule passes if it meets at least 3 of the 4 criteria
        passes_rule_of_five = sum([passes_mw, passes_logp, passes_donors, passes_acceptors]) >= 3

        return {
            "MW <= 500": passes_mw,
            "LogP <= 5": passes_logp,
            "H-bond donors <= 5": passes_donors,
            "H-bond acceptors <= 10": passes_acceptors,
            "Passes Rule of 5": passes_rule_of_five
        }

    @staticmethod
    def calculate_properties(smiles):
        """
        Calculate multiple molecular properties for a SMILES string.

        Args:
            smiles (str): SMILES string

        Returns:
            dict: Dictionary of molecular properties
        """
        mol = Chem.MolFromSmiles(smiles) if smiles else None

        if mol is None:
            return {
                "valid": False,
                "qed": 0.0,
                "sa_score": 10.0,
                "logp": 0.0,
                "molecular_weight": 0.0,
                "rotatable_bonds": 0,
                "lipinski": {
                    "Passes Rule of 5": False
                }
            }

        return {
            "valid": True,
            "qed": MolecularMetrics.calculate_drug_likeness(mol),
            "sa_score": MolecularMetrics.calculate_sa_score(mol),
            "logp": MolecularMetrics.calculate_logp(mol),
            "molecular_weight": MolecularMetrics.calculate_mw(mol),
            "rotatable_bonds": MolecularMetrics.calculate_num_rotatable_bonds(mol),
            "lipinski": MolecularMetrics.check_lipinski_rule_of_five(mol)
        }

    @staticmethod
    def calculate_bulk_properties(smiles_list):
        """
        Calculate properties for a list of SMILES strings.

        Args:
            smiles_list (list): List of SMILES strings

        Returns:
            pd.DataFrame: DataFrame with molecular properties
        """
        properties = []

        for smiles in smiles_list:
            props = MolecularMetrics.calculate_properties(smiles)
            props["smiles"] = smiles
            properties.append(props)

        # Convert to DataFrame
        df = pd.DataFrame(properties)

        # Extract Lipinski data
        lipinski_df = pd.DataFrame([p["lipinski"] for p in properties])

        # Combine DataFrames
        result_df = pd.concat([df.drop("lipinski", axis=1), lipinski_df], axis=1)

        return result_df

    @staticmethod
    def internal_diversity(smiles_list, n_jobs=-1):
        """
        Calculate internal diversity of a set of molecules using Tanimoto similarity.

        Args:
            smiles_list (list): List of SMILES strings
            n_jobs (int): Number of parallel jobs for computation

        Returns:
            float: Internal diversity score
        """
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        import numpy as np
        from joblib import Parallel, delayed

        # Convert SMILES to molecules and compute fingerprints
        valid_mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid_mols = [m for m in valid_mols if m is not None]

        if len(valid_mols) < 2:
            return 0.0

        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in valid_mols]

        # Calculate all pairwise similarities
        n = len(fingerprints)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarities.append(sim)

        # Internal diversity is 1 - average similarity
        avg_similarity = np.mean(similarities)
        internal_diversity = 1 - avg_similarity

        return internal_diversity


def calculate_properties():
    return None


def calculate_validity():
    return None


def calculate_uniqueness():
    return None


def calculate_novelty():
    return None