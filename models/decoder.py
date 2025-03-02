import torch.nn as nn
import math
import torch.nn.functional as F
import torch

class MCDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x):
        # Always apply dropout, regardless of model.train() or model.eval()
        return F.dropout(x, p=self.p, training=True)

class FlexibleDecoder(nn.Module):
    def __init__(
        self,
        total_conv_layers=5,
        total_pool_layers=2,
        initial_channels=64,
        convs_per_block=None,
        channel_growth='double',
        num_classes=1,
        z_dependant_mode=False,
        latent_dim=512,  # Add latent_dim for LUPI
        use_privileged_info=False  # LUPI flag
    ):
        super(FlexibleDecoder, self).__init__()
        self.total_conv_layers = total_conv_layers
        self.total_pool_layers = total_pool_layers
        self.initial_channels = initial_channels
        self.channel_growth = channel_growth
        self.z_dependant_mode = z_dependant_mode
        self.num_classes = num_classes
        self.use_privileged_info = use_privileged_info
        self.latent_dim = latent_dim

        # If convs_per_block is not specified, distribute conv layers evenly
        if convs_per_block is None:
            if total_pool_layers > 0:
                convs_per_block = math.ceil(total_conv_layers / total_pool_layers)
            else:
                convs_per_block = total_conv_layers

        self.features = self._make_layers(convs_per_block)
        
        # Create classifier based on mode
        if z_dependant_mode:
            # Create a more powerful classifier for z-dependent mode
            # Use batch normalization and residual connections
            class ResidualBlock(nn.Module):
                def __init__(self, channels):
                    super(ResidualBlock, self).__init__()
                    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                    self.bn1 = nn.BatchNorm2d(channels)
                    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                    self.bn2 = nn.BatchNorm2d(channels)
                
                def forward(self, x):
                    residual = x
                    out = F.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    out += residual
                    out = F.relu(out)
                    return out
            
            # Replace the features with a better network
            new_features = []
            new_features.extend(list(self.features))
            
            # Add more power with residual blocks at the end
            last_channels = self.final_num_channels
            new_features.append(ResidualBlock(last_channels))
            new_features.append(ResidualBlock(last_channels))
            
            self.features = nn.Sequential(*new_features)
            
            # More powerful classifier head
            # For LUPI, we need to separate the feature extraction part from the final classification
            self.feature_extractor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.BatchNorm1d(self.final_num_channels),
                nn.Dropout(0.3),
                nn.Linear(self.final_num_channels, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5),
                nn.Dropout(0.2),
                nn.BatchNorm1d(128),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256),
            )
            
            # Final classification layer for standard path
            self.final_classifier = nn.Linear(256, num_classes)
            
            # For LUPI: Add a separate branch that processes the privileged z information
            if self.use_privileged_info:
                self.privileged_branch = nn.Sequential(
                    nn.Linear(latent_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2),
                )
                
                # Fusion layer that combines image features with z information
                self.fusion = nn.Sequential(
                    nn.Linear(256 + 256, 256),  # Combine privileged and image features
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.final_num_channels, 1),
                nn.Sigmoid(),
            )

    def _make_layers(self, convs_per_block):
        layers = []
        in_channels = 3  # RGB images
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
                # You can add more channel growth strategies here
                self.final_num_channels = current_channels  # Update final_num_channels

        # Adaptive Average Pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Update the final number of channels after all layers
        self.final_num_channels = in_channels  # in_channels is current_channels at this point

        return nn.Sequential(*layers)

    def forward(self, x, z=None):
        x = self.features(x)
        
        if self.z_dependant_mode:
            if self.use_privileged_info and self.training and z is not None:
                # During training with privileged information
                image_features = self.feature_extractor(x)
                
                # Process z through privileged branch
                z_features = self.privileged_branch(z)
                
                # Concatenate features
                combined_features = torch.cat([image_features, z_features], dim=1)
                
                # Final fusion layer
                return self.fusion(combined_features)
            else:
                # Standard path for inference or when not using privileged info
                image_features = self.feature_extractor(x)
                return self.final_classifier(image_features)
        else:
            # Original binary classification
            x = self.classifier(x)
            return x
