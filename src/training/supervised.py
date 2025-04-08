# src/training/supervised.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
import time

class SupervisedTrainer:
    """
    Trainer class for supervised pre-training of the generator and discriminator.
    """
    def __init__(self, generator, discriminator, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer with models.

        Args:
            generator: Generator model
            discriminator: Discriminator model
            device: Device to run training on
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)

        # Initialize optimizers with default parameters
        self.gen_optimizer = None
        self.disc_optimizer = None

        # Loss functions
        self.gen_criterion = nn.CrossEntropyLoss()
        self.disc_criterion = nn.BCEWithLogitsLoss()

    def setup_optimizers(self, gen_lr=1e-4, disc_lr=1e-4):
        """Setup optimizers with learning rates."""
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=gen_lr)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=disc_lr)

    def train_generator(self, dataloader, epochs=10, save_path=None):
        """
        Train the generator model in supervised mode.

        Args:
            dataloader: DataLoader with SMILES sequences
            epochs: Number of epochs to train
            save_path: Path to save model checkpoints

        Returns:
            List of training losses per epoch
        """
        if self.gen_optimizer is None:
            self.setup_optimizers()

        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Generator Epoch {epoch+1}/{epochs}")

            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                self.gen_optimizer.zero_grad()
                outputs = self.generator(inputs)

                # Reshape outputs and targets for loss calculation
                batch_size, seq_len, vocab_size = outputs.size()
                outputs = outputs.view(-1, vocab_size)
                targets = targets.view(-1)

                # Calculate loss
                loss = self.gen_criterion(outputs, targets)

                # Backward pass
                loss.backward()
                self.gen_optimizer.step()

                # Update progress bar
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss/(batch_idx+1))

            # Record epoch loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            losses.append(avg_epoch_loss)
            print(f"Generator Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

            # Save model checkpoint
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.gen_optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, os.path.join(save_path, f"generator_epoch_{epoch+1}.pt"))

        return losses

    def train_discriminator(self, dataloader, property_data, epochs=10, save_path=None):
        """
        Train the discriminator model to predict molecular properties.

        Args:
            dataloader: DataLoader with SMILES sequences
            property_data: Dictionary mapping SMILES to property values
            epochs: Number of epochs to train
            save_path: Path to save model checkpoints

        Returns:
            List of training losses per epoch
        """
        if self.disc_optimizer is None:
            self.setup_optimizers()

        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Discriminator Epoch {epoch+1}/{epochs}")

            for batch_idx, (smiles_batch, _) in enumerate(progress_bar):
                # Get property values for this batch
                property_values = torch.tensor([property_data.get(smiles, 0.0) for smiles in smiles_batch])

                # Convert SMILES to appropriate input format for discriminator
                smiles_encoded = self.discriminator.encode_smiles_batch(smiles_batch)

                inputs = smiles_encoded.to(self.device)
                targets = property_values.to(self.device).float().unsqueeze(1)

                # Forward pass
                self.disc_optimizer.zero_grad()
                outputs = self.discriminator(inputs)

                # Calculate loss
                loss = self.disc_criterion(outputs, targets)

                # Backward pass
                loss.backward()
                self.disc_optimizer.step()

                # Update progress bar
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss/(batch_idx+1))

            # Record epoch loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            losses.append(avg_epoch_loss)
            print(f"Discriminator Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

            # Save model checkpoint
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.disc_optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, os.path.join(save_path, f"discriminator_epoch_{epoch+1}.pt"))

        return losses


# In training/supervised.py

def pretrain_generator(generator, dataloader, device, epochs=10, learning_rate=1e-4, save_path=None):
    """
    Pretrain generator on existing SMILES data using supervised learning.
    """
    generator.to(device)
    generator.train()

    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is pad token

    history = {'loss': []}

    for epoch in range(epochs):
        epoch_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Get input and target sequences
            input_seq = batch['input'].to(device)
            target_seq = batch['target'].to(device)

            # Forward pass
            optimizer.zero_grad()
            output, _ = generator(input_seq)

            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = output.size()
            output = output.view(-1, vocab_size)
            target_seq = target_seq.view(-1)

            # Calculate loss
            loss = criterion(output, target_seq)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Record and print epoch loss
        avg_loss = epoch_loss / len(dataloader)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model
        if save_path:
            torch.save(generator.state_dict(), save_path)

    return history

def pretrain_discriminator(discriminator, dataloader, device, epochs=10, learning_rate=1e-4, save_path=None):
    """
    Pretrain discriminator to predict drug-likeness of molecules.
    """
    discriminator.to(device)
    discriminator.train()

    optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    bce_criterion = nn.BCELoss()
    mse_criterion = nn.MSELoss()

    history = {'loss': []}

    for epoch in range(epochs):
        epoch_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Get real data
            real_seq = batch['smiles_seq'].to(device)
            real_properties = batch['properties'].to(device) if 'properties' in batch else None

            # Real data labels
            real_labels = torch.ones(real_seq.size(0), 1).to(device)

            # Forward pass with real data
            optimizer.zero_grad()
            real_pred, real_prop_pred = discriminator(real_seq)

            # Calculate loss
            real_loss = bce_criterion(real_pred, real_labels)

            # Property prediction loss (if available)
            prop_loss = 0
            if real_properties is not None:
                prop_loss = mse_criterion(real_prop_pred, real_properties)

            # Total loss
            loss = real_loss + prop_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Record and print epoch loss
        avg_loss = epoch_loss / len(dataloader)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model
        if save_path:
            torch.save(discriminator.state_dict(), save_path)

    return history


