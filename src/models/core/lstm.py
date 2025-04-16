from typing import Optional
import torch
from torch import nn


class LSTM(nn.Module):
    """Core LSTM model implementation.

    This class implements the core LSTM model without any Lightning-specific logic.
    It can be used independently or as part of a Lightning module.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        prediction_length: Number of time steps to predict
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        prediction_length: int = 1,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, prediction_length)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LSTM model.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, num_features]
            
        Returns:
            Output tensor of shape [batch_size, prediction_length, 1]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch_size, sequence_length, hidden_size]
        
        # Get the last hidden state for each sequence
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Predict multiple steps
        predictions = self.fc(last_hidden)  # [batch_size, prediction_length]
        
        # Reshape to match target shape [batch_size, prediction_length, 1]
        predictions = predictions.unsqueeze(-1)
        
        return predictions
