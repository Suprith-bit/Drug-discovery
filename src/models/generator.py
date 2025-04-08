import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MoleculeGenerator(nn.Module):
    """
    Generator network for creating SMILES strings of molecules.
    Uses a GRU-based architecture to generate sequences of characters.
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=3, dropout=0.3):
        """
        Initialize the generator network.

        Args:
            input_size (int): Size of the input embedding (latent dimension)
            hidden_size (int): Size of the GRU hidden state
            output_size (int): Size of the output vocabulary (number of possible characters)
            n_layers (int): Number of GRU layers
            dropout (float): Dropout probability
        """
        super(MoleculeGenerator, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.input_size = input_size

        # Embedding layer to convert one-hot encoded input to dense vectors
        self.embedding = nn.Embedding(output_size, input_size)

        # GRU layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Additional layers for processing latent vector
        self.latent_fc = nn.Linear(input_size, hidden_size)
        self.latent_activation = nn.Tanh()

    def forward(self, input_seq, hidden=None, temperature=1.0):
        """
        Forward pass through the generator.

        Args:
            input_seq (torch.Tensor): Batch of input sequences [batch_size, seq_len]
            hidden (torch.Tensor, optional): Initial hidden state
            temperature (float): Temperature parameter for sampling

        Returns:
            torch.Tensor: Output logits
            torch.Tensor: Hidden state
        """
        # Convert input to embeddings
        embedded = self.embedding(input_seq)  # [batch_size, seq_len, input_size]

        # Initialize hidden state if not provided
        if hidden is None:
            batch_size = input_seq.size(0)
            hidden = self.init_hidden(batch_size)

        # Pass through GRU
        output, hidden = self.gru(embedded, hidden)  # output: [batch_size, seq_len, hidden_size]

        # Pass through output layer
        output = self.fc(output)  # [batch_size, seq_len, output_size]

        # Apply temperature scaling for sampling diversity
        if temperature != 1.0:
            output = output / temperature

        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden state with zeros.

        Args:
            batch_size (int): Batch size

        Returns:
            torch.Tensor: Initialized hidden state
        """
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.embedding.weight.device)

    def init_from_latent(self, latent_vector):
        """
        Initialize hidden state from a latent vector.

        Args:
            latent_vector (torch.Tensor): Latent vector [batch_size, input_size]

        Returns:
            torch.Tensor: Initialized hidden state
        """
        batch_size = latent_vector.size(0)

        # Process latent vector
        h0 = self.latent_activation(self.latent_fc(latent_vector))

        # Expand to all layers
        hidden = h0.unsqueeze(0).expand(self.n_layers, -1, -1).contiguous()

        return hidden

    def sample(self, start_char, end_char, max_length=100, latent=None, device='cpu', temperature=1.0):
        """
        Sample a new SMILES string from the generator.

        Args:
            start_char (int): Index of the start character
            end_char (int): Index of the end character
            max_length (int): Maximum length of the generated sequence
            latent (torch.Tensor, optional): Latent vector to condition generation
            device (str): Device to use for computation
            temperature (float): Temperature parameter for sampling

        Returns:
            list: Sequence of character indices
        """
        with torch.no_grad():
            # Start with batch size of 1
            batch_size = 1 if latent is None else latent.size(0)

            # Initialize hidden state
            if latent is not None:
                hidden = self.init_from_latent(latent)
            else:
                hidden = self.init_hidden(batch_size)

            # Start with start character
            input_seq = torch.tensor([[start_char]] * batch_size, device=device)

            # List to store output
            generated_seq = [start_char]

            # Generate characters until end character or max length
            for i in range(max_length):
                output, hidden = self.forward(input_seq, hidden, temperature)

                # Sample from output distribution
                probs = F.softmax(output[:, -1], dim=-1)
                next_char = torch.multinomial(probs, 1).item()

                # Add to output
                generated_seq.append(next_char)

                # Stop if end character
                if next_char == end_char:
                    break

                # Update input
                input_seq = torch.tensor([[next_char]] * batch_size, device=device)

            return generated_seq

    def generate_smiles(self, vocab, n_samples=1, max_length=100, latent=None, device='cpu', temperature=1.0):
        """
        Generate SMILES strings using the model.

        Args:
            vocab: Vocabulary object with char2idx and idx2char mappings
            n_samples (int): Number of SMILES strings to generate
            max_length (int): Maximum length of the generated sequences
            latent (torch.Tensor, optional): Latent vectors to condition generation
            device (str): Device to use for computation
            temperature (float): Temperature parameter for sampling

        Returns:
            list: List of generated SMILES strings
        """
        start_char = vocab.char2idx['G']  # 'G' for GO token
        end_char = vocab.char2idx['E']    # 'E' for EOS token

        generated_smiles = []

        for _ in range(n_samples):
            # Generate sequence
            seq = self.sample(start_char, end_char, max_length, latent, device, temperature)

            # Convert to string
            smiles = ''.join([vocab.idx2char[idx] for idx in seq[1:-1]])  # Remove start and end tokens
            generated_smiles.append(smiles)

        return generated_smiles


class LatentGenerator(nn.Module):
    """
    Generator that creates latent vectors from random noise.
    Used in the GAN setup to create inputs for the MoleculeGenerator.
    """

    def __init__(self, noise_dim, latent_dim, hidden_dims=[512, 512]):
        """
        Initialize the latent generator.

        Args:
            noise_dim (int): Dimension of the input noise
            latent_dim (int): Dimension of the output latent vector
            hidden_dims (list): Dimensions of hidden layers
        """
        super(LatentGenerator, self).__init__()

        layers = []

        # First layer from noise_dim
        layers.append(nn.Linear(noise_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        layers.append(nn.Tanh())  # Bound outputs to [-1, 1]

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        """
        Forward pass through the generator.

        Args:
            z (torch.Tensor): Random noise [batch_size, noise_dim]

        Returns:
            torch.Tensor: Latent vectors [batch_size, latent_dim]
        """
        return self.model(z)


def get_noise(batch_size, noise_dim, device='cpu'):
    """
    Generate random noise for the generator.

    Args:
        batch_size (int): Batch size
        noise_dim (int): Dimension of the noise
        device (str): Device to use for computation

    Returns:
        torch.Tensor: Random noise [batch_size, noise_dim]
    """
    return torch.randn(batch_size, noise_dim, device=device)


class Generator:
    pass


class SMILESGenerator:
    class SMILESGenerator(nn.Module):
        """
        Generator for creating SMILES strings of molecules.
        """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, n_layers=3, dropout=0.3):
        super(SMILESGenerator, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """Forward pass"""
        # x shape: [batch_size, seq_length]
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # LSTM
        output, hidden = self.lstm(x, hidden)

        # Reshape output for parallel processing of all time steps
        output = output.contiguous().view(-1, self.hidden_dim)

        # Apply output layer
        output = self.fc(output)

        # Reshape back to [batch_size, seq_length, vocab_size]
        output = output.view(batch_size, -1, self.vocab_size)

        return output, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.n_layers, batch_size, self.hidden_dim),
            weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)
        )

    def sample(self, start_char, end_char, max_length=100, device='cpu', temperature=1.0):
        """Sample a new SMILES string"""
        with torch.no_grad():
            # Initialize with start character
            input_seq = torch.tensor([[start_char]], device=device)
            hidden = None

            # Store generated sequence
            generated = [start_char]

            # Generate until end character or max length
            for i in range(max_length - 1):
                # Forward pass
                output, hidden = self.forward(input_seq, hidden)

                # Apply temperature to logits
                output = output[:, -1, :] / temperature

                # Sample from the distribution
                probs = F.softmax(output, dim=-1)
                next_char = torch.multinomial(probs, 1).item()

                # Add to sequence
                generated.append(next_char)

                # Stop if end token is generated
                if next_char == end_char:
                    break

                # Update input for next step
                input_seq = torch.tensor([[next_char]], device=device)

            return generated
    pass