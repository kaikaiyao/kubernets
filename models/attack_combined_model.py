import torch
import torch.nn as nn

from models.decoder import FlexibleDecoder


class TrainableCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(TrainableCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return self.conv5(x)



class CombinedModel(nn.Module):
    def __init__(self, input_channels=3,
                 decoder_total_conv_layers=5, decoder_total_pool_layers=2, 
                 decoder_initial_channels=64):
        """
        Combined model replicating the pipeline of: Image -> Trainable CNN -> Decoder.
        
        Args:
            input_channels (int): Number of input channels for the CNN (e.g. 3 for RGB).
            decoder_total_conv_layers (int): Total conv layers in the decoder.
            decoder_total_pool_layers (int): Total pool layers in the decoder.
            decoder_initial_channels (int): Initial channels for the decoder.
        """
        super(CombinedModel, self).__init__()
        self.cnn = TrainableCNN(input_channels=input_channels, output_channels=input_channels) # Trainable CNN, with same structure to the masked key CNN
        self.decoder = FlexibleDecoder(
            total_conv_layers=decoder_total_conv_layers,
            total_pool_layers=decoder_total_pool_layers,
            initial_channels=decoder_initial_channels
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.decoder(x)
        return x
