#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DrugDiscoveryGAN: Main execution script

This script orchestrates the entire process of drug discovery using a combination
of Generative Adversarial Networks (GANs) and Reinforcement Learning (RL).
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from rdkit import RDLogger
import matplotlib.pyplot as plt
from datetime import datetime

# Custom imports from the project
from data.data_loader import load_dataset, SMILESDataset, get_dataloader
from data.preprocessor import preprocess_smiles_dataset
from models.generator import SMILESGenerator
from models.discriminator import MolecularDiscriminator
from models.gan import DrugGAN
from training.supervised import pretrain_generator, pretrain_discriminator
from training.reinforcement import train_gan_with_reinforcement
from utils.molecular_metrics import calculate_validity, calculate_uniqueness, calculate_novelty
from utils.visualization import MoleculeVisualizer

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DrugDiscoveryGAN')

    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/raw/zinc_clean_leads.csv',
                        help='Path to the SMILES dataset')
    parser.add_argument('--processed_data_path', type=str, default='data/processed/',
                        help='Path to save processed datasets')
    parser.add_argument('--results_path', type=str, default='data/results/',
                        help='Path to save generated molecules and results')

    # Model configuration
    parser.add_argument('--config', type=str, default='configs/model_config.json',
                        help='Path to model configuration file')
    parser.add_argument('--training_config', type=str, default='configs/training_config.json',
                        help='Path to training configuration file')

    # Training parameters
    parser.add_argument('--pretrain_generator', action='store_true',
                        help='Pretrain the generator with supervised learning')
    parser.add_argument('--pretrain_discriminator', action='store_true',
                        help='Pretrain the discriminator with supervised learning')
    parser.add_argument('--train_gan', action='store_true',
                        help='Train the GAN with reinforcement learning')
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate molecules using a pretrained model')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate a set of generated molecules')

    # Model loading/saving
    parser.add_argument('--load_generator', type=str, default=None,
                        help='Path to load a pretrained generator model')
    parser.add_argument('--load_discriminator', type=str, default=None,
                        help='Path to load a pretrained discriminator model')
    parser.add_argument('--load_gan', type=str, default=None,
                        help='Path to load a pretrained GAN model')

    # Generation parameters
    parser.add_argument('--num_molecules', type=int, default=1000,
                        help='Number of molecules to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling during generation')

    # Device selection
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')

    # Experiment name
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment for organizing results')

    # Seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_configs(config_path, training_config_path):
    """Load model and training configurations from JSON files."""
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    with open(training_config_path, 'r') as f:
        training_config = json.load(f)

    return model_config, training_config

def prepare_experiment_directories(args):
    """Prepare directories for experiment results."""
    # Create timestamp for experiment if name not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"experiment_{timestamp}"

    # Create directory structure
    experiment_dir = os.path.join(args.results_path, args.experiment_name)
    models_dir = os.path.join(experiment_dir, "models")
    molecules_dir = os.path.join(experiment_dir, "molecules")
    plots_dir = os.path.join(experiment_dir, "plots")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(molecules_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    return {
        "experiment": experiment_dir,
        "models": models_dir,
        "molecules": molecules_dir,
        "plots": plots_dir
    }

def save_experiment_config(args, experiment_dir):
    """Save experiment configuration for reproducibility."""
    # Convert args to dictionary
    args_dict = vars(args)

    # Save as JSON
    config_path = os.path.join(experiment_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Load model and training configurations
    model_config, training_config = load_configs(args.config, args.training_config)

    # Prepare directories for experiment results
    experiment_dirs = prepare_experiment_directories(args)

    # Save experiment configuration
    save_experiment_config(args, experiment_dirs["experiment"])

    # Load and preprocess the dataset
    print(f"Loading and preprocessing dataset from {args.data_path}...")
    smiles_data = load_dataset(args.data_path)
    processed_data = preprocess_smiles_dataset(smiles_data)

    # Create dataset and dataloader
    dataset = SMILESDataset(processed_data, max_length=model_config["max_seq_length"])
    dataloader = get_dataloader(dataset, batch_size=training_config["batch_size"])

    # Initialize models
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize Generator
    generator = SMILESGenerator(
        vocab_size=dataset.vocab_size,
        embedding_dim=model_config["generator_embedding_dim"],
        hidden_dim=model_config["generator_hidden_dim"],
        n_layers=model_config["generator_n_layers"],
        dropout=model_config["generator_dropout"]
    ).to(device)

    # Initialize Discriminator
    discriminator = MolecularDiscriminator(
        vocab_size=dataset.vocab_size,
        embedding_dim=model_config["discriminator_embedding_dim"],
        hidden_dim=model_config["discriminator_hidden_dim"],
        n_layers=model_config["discriminator_n_layers"],
        n_properties=model_config["n_molecular_properties"],
        dropout=model_config["discriminator_dropout"]
    ).to(device)

    # Load pretrained models if specified
    if args.load_generator:
        print(f"Loading pretrained generator from {args.load_generator}...")
        generator.load_state_dict(torch.load(args.load_generator))

    if args.load_discriminator:
        print(f"Loading pretrained discriminator from {args.load_discriminator}...")
        discriminator.load_state_dict(torch.load(args.load_discriminator))

    # Initialize GAN model
    gan_model = DrugGAN(
        generator=generator,
        discriminator=discriminator,
        char_to_idx=dataset.char_to_idx,
        idx_to_char=dataset.idx_to_char,
        max_seq_length=model_config["max_seq_length"],
        device=device
    )

    # Load pretrained GAN if specified
    if args.load_gan:
        print(f"Loading pretrained GAN from {args.load_gan}...")
        gan_model.load(args.load_gan)

    # Training or generation based on arguments
    if args.pretrain_generator:
        print("Pretraining generator with supervised learning...")
        generator_history = pretrain_generator(
            generator=generator,
            dataloader=dataloader,
            device=device,
            epochs=training_config["generator_epochs"],
            learning_rate=training_config["generator_lr"],
            save_path=os.path.join(experiment_dirs["models"], "pretrained_generator.pt")
        )

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(generator_history['loss'])
        plt.title('Generator Pretraining Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(experiment_dirs["plots"], "generator_pretraining_loss.png"))

    if args.pretrain_discriminator:
        print("Pretraining discriminator with supervised learning...")
        discriminator_history = pretrain_discriminator(
            discriminator=discriminator,
            dataloader=dataloader,
            device=device,
            epochs=training_config["discriminator_epochs"],
            learning_rate=training_config["discriminator_lr"],
            save_path=os.path.join(experiment_dirs["models"], "pretrained_discriminator.pt")
        )

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(discriminator_history['loss'])
        plt.title('Discriminator Pretraining Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(experiment_dirs["plots"], "discriminator_pretraining_loss.png"))

    if args.train_gan:
        print("Training GAN with reinforcement learning...")
        gan_history = train_gan_with_reinforcement(
            gan_model=gan_model,
            dataloader=dataloader,
            epochs=training_config["gan_epochs"],
            generator_lr=training_config["rl_generator_lr"],
            discriminator_lr=training_config["rl_discriminator_lr"],
            n_samples=training_config["n_samples_per_epoch"],
            save_path=os.path.join(experiment_dirs["models"], "trained_gan.pt")
        )

        # Plot training history
        MoleculeVisualizer.plot_training_history(
            gan_history,
            filename=os.path.join(experiment_dirs["plots"], "gan_training_history.png")
        )

    # Generate molecules
    if args.generate_only or args.train_gan:
        print(f"Generating {args.num_molecules} molecules...")
        generated_smiles = gan_model.generate_molecules(
            n_samples=args.num_molecules,
            temperature=args.temperature
        )

        # Save generated molecules
        output_file = os.path.join(experiment_dirs["molecules"], "generated_molecules.csv")
        df = pd.DataFrame({'SMILES': generated_smiles})
        df.to_csv(output_file, index=False)
        print(f"Generated molecules saved to {output_file}")

        # Evaluate generated molecules
        validity = calculate_validity(generated_smiles)
        uniqueness = calculate_uniqueness(generated_smiles)
        novelty = calculate_novelty(generated_smiles, processed_data)

        print(f"Validity: {validity:.2f}")
        print(f"Uniqueness: {uniqueness:.2f}")
        print(f"Novelty: {novelty:.2f}")

        # Save evaluation results
        eval_results = {
            "total_molecules": args.num_molecules,
            "valid_molecules": int(validity * args.num_molecules),
            "unique_molecules": int(uniqueness * int(validity * args.num_molecules)),
            "novel_molecules": int(novelty * int(uniqueness * int(validity * args.num_molecules))),
            "validity": float(validity),
            "uniqueness": float(uniqueness),
            "novelty": float(novelty)
        }

        eval_path = os.path.join(experiment_dirs["molecules"], "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=4)

        # Visualize molecules
        valid_mols = [s for s in generated_smiles if validity > 0]
        if valid_mols:
            # Visualize a random sample of generated molecules
            sample_size = min(20, len(valid_mols))
            sample_idx = np.random.choice(len(valid_mols), sample_size, replace=False)
            sample_smiles = [valid_mols[i] for i in sample_idx]

            MoleculeVisualizer.visualize_molecules(
                sample_smiles,
                filename=os.path.join(experiment_dirs["plots"], "sample_molecules.png")
            )

            # Compare property distributions
            if len(processed_data) > 0 and len(valid_mols) > 0:
                from rdkit.Chem import Descriptors, QED

                # QED (drug-likeness)
                MoleculeVisualizer.plot_property_distribution(
                    valid_mols,
                    processed_data,
                    lambda mol: QED.qed(mol),
                    "QED (Drug-likeness)",
                    filename=os.path.join(experiment_dirs["plots"], "qed_distribution.png")
                )

                # Molecular weight
                MoleculeVisualizer.plot_property_distribution(
                    valid_mols,
                    processed_data,
                    lambda mol: Descriptors.MolWt(mol),
                    "Molecular Weight",
                    filename=os.path.join(experiment_dirs["plots"], "molwt_distribution.png")
                )

                # LogP (lipophilicity)
                MoleculeVisualizer.plot_property_distribution(
                    valid_mols,
                    processed_data,
                    lambda mol: Descriptors.MolLogP(mol),
                    "LogP",
                    filename=os.path.join(experiment_dirs["plots"], "logp_distribution.png")
                )

                # Chemical space visualization
                MoleculeVisualizer.plot_chemical_space(
                    valid_mols,
                    processed_data,
                    n_samples=1000,
                    method='PCA',
                    filename=os.path.join(experiment_dirs["plots"], "chemical_space_pca.png")
                )

if __name__ == "__main__":
    main()