# src/models/gan.py
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from torch.autograd import Variable
import numpy as np
from .generator import Generator
from .discriminator import Discriminator
from ..utils.smiles_utils import decode_smiles
from ..utils.molecular_metrics import calculate_properties

class MolecularGAN(nn.Module):
    """
    Combined GAN model for molecular generation with RL enhancement.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=3, dropout=0.2,
                 property_predictors=None, lr_gen=0.001, lr_dis=0.0005):
        """
        Initialize the GAN model.

        Args:
            vocab_size (int): Size of the SMILES vocabulary
            embedding_dim (int): Dimension of embeddings
            hidden_dim (int): Dimension of hidden layers
            n_layers (int): Number of RNN layers
            dropout (float): Dropout probability
            property_predictors (list): List of property prediction modules for discriminator
            lr_gen (float): Learning rate for generator
            lr_dis (float): Learning rate for discriminator
        """
        super(MolecularGAN, self).__init__()

        # Initialize generator and discriminator
        self.generator = Generator(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
        self.discriminator = Discriminator(vocab_size, embedding_dim, hidden_dim, n_layers,
                                           dropout, property_predictors)

        # Setup optimizers
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr_gen)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_dis)

        # Loss functions
        self.mle_criterion = nn.CrossEntropyLoss(ignore_index=0)  # For pre-training
        self.bce_criterion = nn.BCELoss()  # For adversarial training
        self.mse_criterion = nn.MSELoss()  # For property prediction

        # Training parameters
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def supervised_generator_step(self, input_seqs, target_seqs):
        """
        Training step for supervised learning of generator (Phase 1).

        Args:
            input_seqs (Tensor): Input sequences
            target_seqs (Tensor): Target sequences

        Returns:
            float: Loss value
        """
        self.gen_optimizer.zero_grad()

        # Forward pass
        output = self.generator(input_seqs)
        output = output.view(-1, self.vocab_size)
        target_seqs = target_seqs.view(-1)

        # Calculate loss
        loss = self.mle_criterion(output, target_seqs)

        # Backpropagation
        loss.backward()
        self.gen_optimizer.step()

        return loss.item()

    def supervised_discriminator_step(self, seqs, property_values):
        """
        Training step for supervised learning of discriminator (Phase 1).

        Args:
            seqs (Tensor): SMILES sequences
            property_values (dict): Dictionary of property values

        Returns:
            float: Loss value
        """
        self.dis_optimizer.zero_grad()

        # Forward pass
        pred_properties = self.discriminator(seqs)

        # Calculate combined loss for all properties
        loss = 0
        for prop_name, pred_value in pred_properties.items():
            if prop_name in property_values:
                loss += self.mse_criterion(pred_value, property_values[prop_name])

        # Backpropagation
        loss.backward()
        self.dis_optimizer.step()

        return loss.item()

    def reinforcement_step(self, batch_size, max_length, temperature=1.0, gamma=0.97):
        """
        Training step for reinforcement learning (Phase 2).

        Args:
            batch_size (int): Batch size
            max_length (int): Maximum sequence length
            temperature (float): Sampling temperature
            gamma (float): Discount factor for rewards

        Returns:
            dict: Dictionary with losses and metrics
        """
        # Generate molecules
        generated_seqs, log_probs = self.generator.sample(batch_size, max_length, temperature)

        # Convert to SMILES
        smiles_list = []
        valid_indices = []
        for i, seq in enumerate(generated_seqs):
            smiles = decode_smiles(seq, self.generator.idx_to_char)
            if smiles is not None:
                smiles_list.append(smiles)
                valid_indices.append(i)

        # Skip if no valid molecules were generated
        if len(smiles_list) == 0:
            return {"gen_loss": 0, "valid_ratio": 0, "unique_ratio": 0, "metrics": {}}

        # Calculate molecular properties
        properties = calculate_properties(smiles_list)

        # Get rewards from discriminator
        valid_seqs = generated_seqs[valid_indices]
        rewards = self.discriminator.get_rewards(valid_seqs, properties)

        # Filter log probs for valid sequences
        valid_log_probs = [log_probs[i] for i in valid_indices]

        # Reinforcement learning for generator
        self.gen_optimizer.zero_grad()

        # Calculate policy gradient loss
        gen_loss = 0
        for log_p, reward in zip(valid_log_probs, rewards):
            # Negative sign because we're maximizing reward
            gen_loss -= log_p * reward

        # Backpropagation
        gen_loss = gen_loss / len(valid_indices) if valid_indices else 0
        if isinstance(gen_loss, torch.Tensor) and gen_loss.requires_grad:
            gen_loss.backward()
            self.gen_optimizer.step()

        # Calculate metrics
        valid_ratio = len(valid_indices) / batch_size
        unique_ratio = len(set(smiles_list)) / len(smiles_list) if smiles_list else 0

        return {
            "gen_loss": gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss,
            "valid_ratio": valid_ratio,
            "unique_ratio": unique_ratio,
            "metrics": properties
        }

    def save(self, path):
        """Save model to disk."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'dis_optimizer_state_dict': self.dis_optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model from disk."""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])


class DrugGAN:
    class DrugGAN:
        """
        Integrated model combining generator and discriminator with reinforcement learning.
        """
    def __init__(self, generator, discriminator, char_to_idx, idx_to_char, max_seq_length=100, device='cpu'):
        self.generator = generator
        self.discriminator = discriminator
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.max_seq_length = max_seq_length
        self.device = device

        # Move models to device
        self.generator.to(device)
        self.discriminator.to(device)

        # Optimizers
        self.gen_optimizer = None
        self.dis_optimizer = None

    def setup_optimizers(self, gen_lr=1e-4, dis_lr=5e-5):
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gen_lr)
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=dis_lr)

    def generate_molecules(self, n_samples=100, temperature=1.0):
        """Generate new molecules"""
        self.generator.eval()

        generated_smiles = []

        for _ in range(n_samples):
            # Generate sequence of token indices
            start_token = self.char_to_idx['<start>'] if '<start>' in self.char_to_idx else 1
            sequence = self.generator.sample(
                start_char=start_token,
                end_char=self.char_to_idx['<end>'] if '<end>' in self.char_to_idx else 2,
                max_length=self.max_seq_length,
                device=self.device,
                temperature=temperature
            )

            # Convert to SMILES
            smiles = ''.join([self.idx_to_char[idx] for idx in sequence[1:-1]])  # Remove start/end tokens

            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                generated_smiles.append(canonical_smiles)

        return generated_smiles

    def save(self, path):
        """Save model to disk"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict() if self.gen_optimizer else None,
            'dis_optimizer_state_dict': self.dis_optimizer.state_dict() if self.dis_optimizer else None,
        }, path)

    def load(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        if checkpoint['gen_optimizer_state_dict'] and self.gen_optimizer:
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        if checkpoint['dis_optimizer_state_dict'] and self.dis_optimizer:
            self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
    pass