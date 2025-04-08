import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MolecularCNN(nn.Module):
    """
    Convolutional neural network for processing SMILES strings as character sequences.
    Used as feature extractor in the discriminator.
    """

    def __init__(self, vocab_size, embedding_dim=128, filters=[3, 4, 5], num_filters=64):
        """
        Initialize the CNN for SMILES processing.

        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of character embeddings
            filters (list): List of filter sizes for CNN
            num_filters (int): Number of filters per size
        """
        super(MolecularCNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, filter_size)
            for filter_size in filters
        ])

        # Output dimension after CNN
        self.output_dim = num_filters * len(filters)

    def forward(self, x):
        """
        Forward pass through the CNN.

        Args:
            x (torch.Tensor): Batch of input sequences [batch_size, seq_len]

        Returns:
            torch.Tensor: Feature representation [batch_size, output_dim]
        """
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # Prepare for 1D convolution (batch, channels, seq_len)
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]

        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, seq_len - filter_size + 1]

            # Max pooling over time
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]

        # Concatenate output from different filter sizes
        x = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filters)]

        return x


class PropertyPredictor(nn.Module):
    """
    Neural network for predicting molecular properties.
    Takes feature representation from CNN and outputs property scores.
    """

    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=1, dropout=0.2):
        """
        Initialize the property predictor.

        Args:
            input_dim (int): Dimension of input features
            hidden_dims (list): Dimensions of hidden layers
            output_dim (int): Number of properties to predict
            dropout (float): Dropout probability
        """
        super(PropertyPredictor, self).__init__()

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the property predictor.

        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]

        Returns:
            torch.Tensor: Predicted properties [batch_size, output_dim]
        """
        return self.model(x)


class Discriminator(nn.Module):
    """
    Discriminator network for evaluating molecules.
    Combines feature extraction with property prediction.
    In the GAN context, this serves as both discriminator and critic.
    """

    def __init__(self, vocab_size, embedding_dim=128, cnn_filters=[3, 4, 5], num_filters=64,
                 hidden_dims=[256, 128], n_properties=1, dropout=0.2):
        """
        Initialize the discriminator.

        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of character embeddings
            cnn_filters (list): List of filter sizes for CNN
            num_filters (int): Number of filters per size
            hidden_dims (list): Dimensions of hidden layers in property predictor
            n_properties (int): Number of properties to predict
            dropout (float): Dropout probability
        """
        super(Discriminator, self).__init__()

        # Feature extractor
        self.feature_extractor = MolecularCNN(
            vocab_size, embedding_dim, cnn_filters, num_filters
        )

        # Property predictor
        self.property_predictor = PropertyPredictor(
            self.feature_extractor.output_dim, hidden_dims, n_properties, dropout
        )

        # Additional head for determining if molecule is real (for GAN training)
        self.real_fake_head = nn.Sequential(
            nn.Linear(self.feature_extractor.output_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Batch of input sequences [batch_size, seq_len]

        Returns:
            tuple: (real_fake_score, property_predictions)
        """
        # Extract features
        features = self.feature_extractor(x)

        # Predict if molecule is real or fake (for GAN)
        real_fake = self.real_fake_head(features)

        # Predict properties
        properties = self.property_predictor(features)

        return real_fake, properties

    def predict_properties(self, x):
        """
        Predict only the properties of molecules.

        Args:
            x (torch.Tensor): Batch of input sequences [batch_size, seq_len]

        Returns:
            torch.Tensor: Predicted properties [batch_size, n_properties]
        """
        features = self.feature_extractor(x)
        return self.property_predictor(features)


class MultiPropertyDiscriminator(nn.Module):
    """
    Extended discriminator for predicting multiple molecular properties.
    Used in the reinforcement learning phase to provide more detailed rewards.
    """

    def __init__(self, vocab_size, embedding_dim=128, cnn_filters=[3, 4, 5], num_filters=64,
                 hidden_dims=[256, 128], property_names=None, dropout=0.2):
        """
        Initialize the multi-property discriminator.

        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of character embeddings
            cnn_filters (list): List of filter sizes for CNN
            num_filters (int): Number of filters per size
            hidden_dims (list): Dimensions of hidden layers in property predictor
            property_names (list): Names of properties to predict
            dropout (float): Dropout probability
        """
        super(MultiPropertyDiscriminator, self).__init__()

        # Set default property names if not provided
        if property_names is None:
            property_names = ['druglikeness', 'solubility', 'synthesizability', 'novelty']

        self.property_names = property_names
        n_properties = len(property_names)

        # Feature extractor
        self.feature_extractor = MolecularCNN(
            vocab_size, embedding_dim, cnn_filters, num_filters
        )

        # Real/fake discriminator
        self.real_fake_head = nn.Sequential(
            nn.Linear(self.feature_extractor.output_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 2, 1),
            nn.Sigmoid()
        )

        # Separate predictors for each property
        self.property_predictors = nn.ModuleDict()

        for prop_name in property_names:
            self.property_predictors[prop_name] = PropertyPredictor(
                self.feature_extractor.output_dim, hidden_dims, 1, dropout
            )

    def forward(self, x):
        """
        Forward pass through the multi-property discriminator.

        Args:
            x (torch.Tensor): Batch of input sequences [batch_size, seq_len]

        Returns:
            tuple: (real_fake_score, property_dict)
        """
        # Extract features
        features = self.feature_extractor(x)

        # Predict if molecule is real or fake
        real_fake = self.real_fake_head(features)

        # Predict each property
        properties = {}
        for prop_name in self.property_names:
            properties[prop_name] = self.property_predictors[prop_name](features)

        return real_fake, properties

    def predict_properties(self, x):
        """
        Predict only the properties of molecules.

        Args:
            x (torch.Tensor): Batch of input sequences [batch_size, seq_len]

        Returns:
            dict: Dictionary of predicted properties
        """
        features = self.feature_extractor(x)

        properties = {}
        for prop_name in self.property_names:
            properties[prop_name] = self.property_predictors[prop_name](features)

        return properties

    def calculate_reward(self, x, reward_weights=None):
        """
        Calculate a weighted reward based on multiple properties.
        Used in reinforcement learning phase.

        Args:
            x (torch.Tensor): Batch of input sequences [batch_size, seq_len]
            reward_weights (dict): Dictionary of weights for each property

        Returns:
            torch.Tensor: Weighted reward [batch_size, 1]
        """
        # Default weights if not provided
        if reward_weights is None:
            reward_weights = {prop: 1.0 / len(self.property_names) for prop in self.property_names}

        # Get all properties
        properties = self.predict_properties(x)

        # Calculate weighted sum
        reward = torch.zeros(x.size(0), 1, device=x.device)
        for prop_name, prop_value in properties.items():
            if prop_name in reward_weights:
                reward += reward_weights[prop_name] * prop_value

        return reward


class MolecularDiscriminator(nn.Module):
    """
    Discriminator for evaluating the drug-likeness of molecules.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2,
                 n_properties=1, dropout=0.2):
        super(MolecularDiscriminator, self).__init__()

        # Feature extractor
        self.feature_extractor = MolecularCNN(
            vocab_size, embedding_dim=embedding_dim, filters=[3, 4, 5], num_filters=64
        )

        # Property prediction head
        hidden_dims = [hidden_dim, hidden_dim//2]
        self.property_predictor = PropertyPredictor(
            self.feature_extractor.output_dim, hidden_dims, n_properties, dropout
        )

        # Real/fake classification head
        self.real_fake_head = nn.Sequential(
            nn.Linear(self.feature_extractor.output_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        real_fake = self.real_fake_head(features)
        properties = self.property_predictor(features)

        return real_fake, properties

    def predict_properties(self, x):
        features = self.feature_extractor(x)
        return self.property_predictor(features)