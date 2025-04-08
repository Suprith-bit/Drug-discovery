import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
import os

class MoleculeVisualizer:
    """
    Class for visualizing molecules and their properties.
    """

    @staticmethod
    def visualize_molecules(smiles_list, n_mols=10, labels=None, filename=None):
        """
        Visualize a list of molecules from SMILES strings.

        Args:
            smiles_list (list): List of SMILES strings to visualize
            n_mols (int): Number of molecules to display
            labels (list, optional): Labels for each molecule
            filename (str, optional): If provided, save the visualization to this file

        Returns:
            PIL.Image: Image containing the molecule grid
        """
        # Convert SMILES to RDKit molecules
        valid_mols = []
        valid_labels = []

        for i, smiles in enumerate(smiles_list[:n_mols]):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                if labels is not None:
                    valid_labels.append(labels[i])

        if not valid_mols:
            print("No valid molecules to visualize.")
            return None

        # Create a molecule grid
        if labels is not None:
            img = Draw.MolsToGridImage(valid_mols, molsPerRow=5, subImgSize=(300, 300), legends=valid_labels)
        else:
            img = Draw.MolsToGridImage(valid_mols, molsPerRow=5, subImgSize=(300, 300))

        # Save the image if filename is provided
        if filename:
            img.save(filename)

        return img

    @staticmethod
    def plot_property_distribution(generated_smiles, reference_smiles, property_fn, property_name, filename=None):
        """
        Create violin plots comparing property distributions between generated and reference molecules.

        Args:
            generated_smiles (list): List of generated SMILES strings
            reference_smiles (list): List of reference SMILES strings
            property_fn (function): Function that calculates the property for a molecule
            property_name (str): Name of the property for axis label
            filename (str, optional): If provided, save the plot to this file
        """
        # Calculate properties for valid molecules
        gen_values = []
        ref_values = []

        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    value = property_fn(mol)
                    if value is not None and not np.isnan(value):
                        gen_values.append(value)
                except:
                    continue

        for smiles in reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    value = property_fn(mol)
                    if value is not None and not np.isnan(value):
                        ref_values.append(value)
                except:
                    continue

        # Create DataFrame for seaborn
        gen_df = pd.DataFrame({property_name: gen_values, 'Source': 'Generated'})
        ref_df = pd.DataFrame({property_name: ref_values, 'Source': 'Reference'})
        df = pd.concat([gen_df, ref_df])

        # Create violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Source', y=property_name, data=df, palette={"Generated": "blue", "Reference": "yellow"})
        plt.title(f'Distribution of {property_name}')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the plot if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_training_history(history, filename=None):
        """
        Plot training metrics history.

        Args:
            history (dict): Dictionary containing training metrics
            filename (str, optional): If provided, save the plot to this file
        """
        plt.figure(figsize=(15, 10))

        # Plot generator and discriminator losses
        plt.subplot(2, 2, 1)
        plt.plot(history['generator_loss'], label='Generator Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history['discriminator_loss'], label='Discriminator Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss')
        plt.legend()

        # Plot validity and uniqueness if available
        if 'validity' in history:
            plt.subplot(2, 2, 3)
            plt.plot(history['validity'], label='Validity')
            if 'uniqueness' in history:
                plt.plot(history['uniqueness'], label='Uniqueness')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Rate')
            plt.title('Molecule Quality Metrics')
            plt.legend()

        # Plot reward if available
        if 'reward' in history:
            plt.subplot(2, 2, 4)
            plt.plot(history['reward'], label='Average Reward')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            plt.title('Reinforcement Learning Reward')
            plt.legend()

        plt.tight_layout()

        # Save the plot if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_chemical_space(generated_smiles, reference_smiles, n_samples=1000, method='PCA', filename=None):
        """
        Visualize the chemical space of generated vs reference molecules using dimensionality reduction.

        Args:
            generated_smiles (list): List of generated SMILES strings
            reference_smiles (list): List of reference SMILES strings
            n_samples (int): Number of molecules to sample for visualization
            method (str): Dimensionality reduction method ('PCA' or 'TSNE')
            filename (str, optional): If provided, save the plot to this file
        """
        # Function to calculate Morgan fingerprints
        def calculate_fingerprints(smiles_list, n_samples):
            fingerprints = []
            valid_smiles = []

            # Randomly sample if there are too many molecules
            if len(smiles_list) > n_samples:
                indices = np.random.choice(len(smiles_list), n_samples, replace=False)
                smiles_sample = [smiles_list[i] for i in indices]
            else:
                smiles_sample = smiles_list

            # Calculate fingerprints for valid molecules
            for smiles in smiles_sample:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    fingerprints.append(np.array(fp))
                    valid_smiles.append(smiles)

            return np.array(fingerprints), valid_smiles

        # Calculate fingerprints for both sets
        gen_fps, gen_valid = calculate_fingerprints(generated_smiles, min(n_samples, len(generated_smiles)))
        ref_fps, ref_valid = calculate_fingerprints(reference_smiles, min(n_samples, len(reference_smiles)))

        if len(gen_fps) == 0 or len(ref_fps) == 0:
            print("Not enough valid molecules to visualize chemical space.")
            return

        # Combine fingerprints for dimensionality reduction
        all_fps = np.vstack([gen_fps, ref_fps])

        # Apply dimensionality reduction
        if method == 'PCA':
            reducer = PCA(n_components=2)
        else:  # TSNE
            reducer = TSNE(n_components=2, random_state=42)

        reduced_fps = reducer.fit_transform(all_fps)

        # Split back into generated and reference
        gen_reduced = reduced_fps[:len(gen_fps)]
        ref_reduced = reduced_fps[len(gen_fps):]

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(ref_reduced[:, 0], ref_reduced[:, 1], alpha=0.5, s=10, c='yellow', label='Reference')
        plt.scatter(gen_reduced[:, 0], gen_reduced[:, 1], alpha=0.5, s=10, c='blue', label='Generated')

        plt.xlabel(f'{method} Component 1')
        plt.ylabel(f'{method} Component 2')
        plt.title(f'Chemical Space Visualization using {method}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the plot if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_scaffolds_comparison(generated_smiles, reference_smiles, top_n=10, filename=None):
        """
        Compare the distribution of molecular scaffolds between generated and reference sets.

        Args:
            generated_smiles (list): List of generated SMILES strings
            reference_smiles (list): List of reference SMILES strings
            top_n (int): Number of top scaffolds to display
            filename (str, optional): If provided, save the plot to this file
        """
        # Function to extract scaffolds
        def get_scaffold_counts(smiles_list):
            scaffolds = {}
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    try:
                        scaffold = MurckoScaffold.GetScaffoldSmiles(mol)
                        if scaffold in scaffolds:
                            scaffolds[scaffold] += 1
                        else:
                            scaffolds[scaffold] = 1
                    except:
                        continue

            # Sort by count
            sorted_scaffolds = sorted(scaffolds.items(), key=lambda x: x[1], reverse=True)
            return sorted_scaffolds

        # Get scaffold counts
        gen_scaffolds = get_scaffold_counts(generated_smiles)
        ref_scaffolds = get_scaffold_counts(reference_smiles)

        # Prepare data for plotting
        top_gen_scaffolds = gen_scaffolds[:top_n]
        scaffold_smiles = [s[0] for s in top_gen_scaffolds]
        gen_counts = [s[1] for s in top_gen_scaffolds]

        # Get counts for the same scaffolds in reference set
        ref_counts = []
        ref_dict = dict(ref_scaffolds)
        for scaffold in scaffold_smiles:
            ref_counts.append(ref_dict.get(scaffold, 0))

        # Convert to percentages
        gen_percent = [count / len(generated_smiles) * 100 for count in gen_counts]
        ref_percent = [count / len(reference_smiles) * 100 for count in ref_counts]

        # Create molecules for visualization
        mols = [Chem.MolFromSmiles(s) for s in scaffold_smiles]
        valid_mols = [m for m in mols if m is not None]

        # Create molecule images
        img = Draw.MolsToGridImage(valid_mols, molsPerRow=5, subImgSize=(200, 200))

        # Plot the comparison
        plt.figure(figsize=(12, 8))
        x = np.arange(len(scaffold_smiles))
        width = 0.35

        plt.bar(x - width/2, gen_percent, width, label='Generated')
        plt.bar(x + width/2, ref_percent, width, label='Reference')

        plt.xlabel('Scaffold Index')
        plt.ylabel('Percentage (%)')
        plt.title('Top Scaffolds Distribution Comparison')
        plt.xticks(x, [f'S{i+1}' for i in range(len(scaffold_smiles))])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the plot if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            img.save(filename.replace('.png', '_structures.png'))

        plt.show()

        # Display scaffold structures
        return img