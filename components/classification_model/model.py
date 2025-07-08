"""This module contains a class for Pytorch model."""
import torch
from torch import nn


class FFNN(nn.Module):
    """This class contains a feedforward neural network for text classification.

    Attributes:
        fc1: First fully connected layer.
        fc2: Second fully connected layer.
        relu: ReLU activation function to use between layers.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Creates an instance of the class.

        Args:
            input_dim: Dimension of input data.
            hidden_dim: Dimension of hidden layer.
            output_dim: Number of outputs.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass on a provided data.

        Args:
            x: Input vector.

        Returns:
            Logits for the input vector.
        """
        return self.fc2(self.relu(self.fc1(x)))
