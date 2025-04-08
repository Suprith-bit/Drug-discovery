
# src/training/reinforcement.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rdkit.Chem import QED, Descriptors
from tqdm import tqdm
import os
import json
import time
from rdkit import Chem

from src.utils.molecular_metrics import MolecularMetrics, calculate_uniqueness, calculate_validity


class RLTrainer:
    """
    Trainer class for reinforcement learning phase where the generator is trained
    with rewards from the discriminator.
    """
    def __init__(self, generator, discriminator,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 char_dict=None):
        """
        Initialize the RL trainer.

        Args:
            generator: Generator model
            discriminator: Discriminator model
            device: Device to run training on
            char_dict: Character to index mapping dictionary
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)
        self.char_dict = char_dict

        # Set discriminator to evaluation mode since we won't be updating it
        self.discriminator.eval()

        # Initialize optimizers with default parameters
        self.gen_optimizer = None

        # Track rewards
        self.reward_history = []
        self.valid_mol_history = []

    def setup_optimizer(self, gen_lr=5e-5):
        """Setup optimizer with learning rate."""
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=gen_lr)

    def calculate_rewards(self, smiles_list):
        """
        Calculate rewards for a batch of generated SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Tensor of rewards
        """
        # Filter valid molecules first
        valid_mols = []
        valid_smiles = []
        invalid_count = 0

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                valid_smiles.append(smiles)
            else:
                invalid_count += 1

        # If no valid molecules, return zero rewards
        if len(valid_smiles) == 0:
            return torch.zeros(len(smiles_list)).to(self.device), 0

        # Convert valid SMILES to appropriate input format for discriminator
        smiles_encoded = self.discriminator.encode_smiles_batch(valid_smiles)
        inputs = smiles_encoded.to(self.device)

        # Get property predictions from discriminator
        with torch.no_grad():
            rewards = self.discriminator(inputs)

        # Convert to probabilities
        rewards = torch.sigmoid(rewards).squeeze()

        # Create full rewards tensor with zeros for invalid molecules
        full_rewards = torch.zeros(len(smiles_list)).to(self.device)
        valid_indices = [i for i, smiles in enumerate(smiles_list) if Chem.MolFromSmiles(smiles) is not None]
        full_rewards[valid_indices] = rewards

        valid_ratio = len(valid_smiles) / len(smiles_list)
        return full_rewards, valid_ratio

    def train_step(self, batch_size=64, temperature=1.0):
        """
        Perform a single training step using policy gradient.

        Args:
            batch_size: Number of molecules to generate per step
            temperature: Sampling temperature for generation

        Returns:
            Dictionary with training metrics
        """
        self.generator.train()

        # Generate molecules
        generated_data = self.generator.sample(batch_size, temperature=temperature)

        # Extract SMILES strings and log probabilities
        smiles_list = generated_data['smiles']
        log_probs = generated_data['log_probs'].to(self.device)

        # Calculate rewards
        rewards, valid_ratio = self.calculate_rewards(smiles_list)

        # Normalize rewards to help stabilize training
        if len(rewards) > 1 and rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Calculate reinforcement learning loss
        rl_loss = -torch.mean(log_probs * rewards)

        # Update generator
        self.gen_optimizer.zero_grad()
        rl_loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0)
        self.gen_optimizer.step()

        # Calculate mean reward for reporting
        mean_reward = rewards.mean().item()

        metrics = {
            'rl_loss': rl_loss.item(),
            'mean_reward': mean_reward,
            'valid_ratio': valid_ratio
        }

        return metrics

    def train(self, epochs=100, steps_per_epoch=100, batch_size=64,
              temperature=1.0, save_path=None, save_interval=10):
        """
        Train the generator using reinforcement learning.

        Args:
            epochs: Number of epochs to train
            steps_per_epoch: Number of batches per epoch
            batch_size: Batch size for generation
            temperature: Temperature for sampling
            save_path: Path to save model checkpoints
            save_interval: Interval for saving checkpoints

        Returns:
            Dictionary of training history
        """
        if self.gen_optimizer is None:
            self.setup_optimizer()

        history = {
            'rl_loss': [],
            'mean_reward': [],
            'valid_ratio': []
        }

        for epoch in range(epochs):
            epoch_metrics = {
                'rl_loss': 0,
                'mean_reward': 0,
                'valid_ratio': 0
            }

            progress_bar = tqdm(range(steps_per_epoch), desc=f"RL Epoch {epoch+1}/{epochs}")

            for step in progress_bar:
                # Perform a training step
                metrics = self.train_step(batch_size=batch_size, temperature=temperature)

                # Update epoch metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key] / steps_per_epoch

                # Update progress bar
                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} - " +
                  f"Loss: {epoch_metrics['rl_loss']:.4f}, " +
                  f"Reward: {epoch_metrics['mean_reward']:.4f}, " +
                  f"Valid: {epoch_metrics['valid_ratio']:.2%}")

            # Update history
            for key in history:
                history[key].append(epoch_metrics[key])

            # Save model
            if save_path and (epoch + 1) % save_interval == 0:
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.gen_optimizer.state_dict(),
                    'metrics': epoch_metrics,
                }, os.path.join(save_path, f"generator_rl_epoch_{epoch+1}.pt"))

                # Save generated molecules for evaluation
                self.generator.eval()
                with torch.no_grad():
                    generated_data = self.generator.sample(100, temperature=temperature)
                    valid_mols = [s for s in generated_data['smiles'] if Chem.MolFromSmiles(s) is not None]

                with open(os.path.join(save_path, f"generated_molecules_epoch_{epoch+1}.json"), 'w') as f:
                    json.dump({
                        'epoch': epoch + 1,
                        'valid_ratio': len(valid_mols) / 100,
                        'molecules': valid_mols
                    }, f, indent=2)

        return history


# In training/reinforcement.py

def train_gan_with_reinforcement(gan_model, dataloader, epochs=50, generator_lr=1e-5,
                                 discriminator_lr=5e-6, n_samples=1000, save_path=None):
    """
    Train GAN using reinforcement learning for molecule generation.
    """
    # Set up optimizers
    if gan_model.gen_optimizer is None or gan_model.dis_optimizer is None:
        gan_model.setup_optimizers(gen_lr=generator_lr, dis_lr=discriminator_lr)

    # Define reward functions
    def get_rewards(smiles_list):
        # Convert to RDKit molecules
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid_indices = [i for i, mol in enumerate(mols) if mol is not None]

        # No valid molecules found
        if not valid_indices:
            return torch.zeros(len(smiles_list), device=gan_model.device)

        # Calculate rewards for valid molecules
        rewards = torch.zeros(len(smiles_list), device=gan_model.device)

        for i in valid_indices:
            mol = mols[i]
            # Calculate drug likeness using QED
            qed_score = QED.qed(mol)
            # Logp in optimal range (0-5)
            logp = Descriptors.MolLogP(mol)
            logp_score = max(0, min(1, 1 - abs(logp - 2.5) / 2.5))
            # Synthetic accessibility (normalized)
            sa_score = max(0, min(1, 1 - MolecularMetrics.calculate_sa_score(mol) / 10))

            # Combined reward
            reward = 0.4 * qed_score + 0.4 * logp_score + 0.2 * sa_score
            rewards[i] = reward

        return rewards

    # History dictionary
    history = {
        'generator_loss': [],
        'discriminator_loss': [],
        'validity': [],
        'uniqueness': [],
        'reward': []
    }

    # Training loop
    for epoch in range(epochs):
        # Generate molecules
        gan_model.generator.train()

        # Sample new molecules
        samples = []

        # Generate in batches to avoid memory issues
        batch_size = 100
        n_batches = n_samples // batch_size

        for _ in tqdm(range(n_batches), desc=f"Generating molecules (Epoch {epoch+1}/{epochs})"):
            batch_samples = gan_model.generate_molecules(n_samples=batch_size)
            samples.extend(batch_samples)

        # Calculate rewards
        rewards = get_rewards([s for s in samples if s])

        # Update generator with policy gradient
        gan_model.gen_optimizer.zero_grad()

        # Calculate policy gradient loss
        # This is a placeholder - the actual implementation would depend on how
        # you track log probabilities during sampling

        # Calculate generator metrics
        valid_ratio = calculate_validity(samples)
        unique_ratio = calculate_uniqueness(samples)

        # Record metrics
        history['validity'].append(valid_ratio)
        history['uniqueness'].append(unique_ratio)
        history['reward'].append(rewards.mean().item() if len(rewards) > 0 else 0)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Validity: {valid_ratio:.4f}, Uniqueness: {unique_ratio:.4f}")
        print(f"Avg Reward: {history['reward'][-1]:.4f}")

        # Save model
        if save_path and (epoch + 1) % 5 == 0:
            gan_model.save(save_path)

    return history