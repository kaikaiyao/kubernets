import torch
import torch.nn as nn
import math

###########################################################
# Define the CNN architecture (similar to RandomCNN, but now trainable)
###########################################################
class TrainableCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(TrainableCNN, self).__init__()
        # Same architecture as before, but we do NOT freeze parameters
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # Note: no normalization or thresholding here, attacker can decide how to handle inputs
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        return x

###########################################################
# We assume the FlexibleDecoder is defined as provided by you:
#
# class FlexibleDecoder(nn.Module):
#     def __init__(...):
#         ...
#     def forward(self, x):
#         ...
###########################################################

# We take the same FlexibleDecoder code from your snippet:
class FlexibleDecoder(nn.Module):
    def __init__(
        self,
        total_conv_layers=5,
        total_pool_layers=2,
        initial_channels=64,
        convs_per_block=None,
        channel_growth='double'
    ):
        super(FlexibleDecoder, self).__init__()
        self.total_conv_layers = total_conv_layers
        self.total_pool_layers = total_pool_layers
        self.initial_channels = initial_channels
        self.channel_growth = channel_growth

        # If convs_per_block is not specified, distribute conv layers evenly
        if convs_per_block is None:
            if total_pool_layers > 0:
                convs_per_block = math.ceil(total_conv_layers / total_pool_layers)
            else:
                convs_per_block = total_conv_layers

        self.features = self._make_layers(convs_per_block)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_num_channels, 1),
            nn.Sigmoid(),
        )

    def _make_layers(self, convs_per_block):
        layers = []
        in_channels = 3  # Decoder expects 3-channel input by default
        current_channels = self.initial_channels
        conv_layers_remaining = self.total_conv_layers
        pool_layers_remaining = self.total_pool_layers

        # Keep track of the final number of channels
        self.final_num_channels = current_channels

        while conv_layers_remaining > 0:
            # Number of conv layers in this block
            convs_in_this_block = min(convs_per_block, conv_layers_remaining)

            # Add convolutional layers
            for _ in range(convs_in_this_block):
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        current_channels,
                        kernel_size=3,
                        padding=1
                    )
                )
                layers.append(nn.ReLU(inplace=True))
                in_channels = current_channels
                conv_layers_remaining -= 1

            # Add pooling layer if any remaining
            if pool_layers_remaining > 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                pool_layers_remaining -= 1

            # Update channels for next block
            if conv_layers_remaining > 0:  # Only update channels if more conv layers are left
                if self.channel_growth == 'double':
                    current_channels *= 2
                elif self.channel_growth == 'linear':
                    current_channels += self.initial_channels
                self.final_num_channels = current_channels  # Update final_num_channels

        # Adaptive Average Pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Update the final number of channels after all layers
        self.final_num_channels = in_channels  # in_channels is current_channels at this point

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


###########################################################
# Define the CombinedModel that attacker trains
###########################################################
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
        # Trainable CNN similar to the masked key CNN but now parameters are not frozen
        self.cnn = TrainableCNN(input_channels=input_channels, output_channels=input_channels)

        # The decoder expects a 3-channel input by default, so ensure that:
        # After the CNN, we get something that matches the decoder input format.
        # By default, the CNN outputs `input_channels` channels.
        # If that doesn't match 3, we might need a small adjustment:
        # For simplicity, let's assume the CNN outputs 3 channels for the decoder.
        # (Already done above: output_channels=input_channels=3)

        self.decoder = FlexibleDecoder(
            total_conv_layers=decoder_total_conv_layers,
            total_pool_layers=decoder_total_pool_layers,
            initial_channels=decoder_initial_channels
        )

    def forward(self, x):
        # Pass through CNN
        x = self.cnn(x)
        # Pass through Decoder
        x = self.decoder(x)
        return x
