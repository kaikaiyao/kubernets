import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import random

class MCDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x):
        # Always apply dropout, regardless of model.train() or model.eval()
        return F.dropout(x, p=self.p, training=True)

class TemperatureScaledSigmoid(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x):
        return torch.sigmoid(x / self.temperature)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = MCDropout(p=0.1)  # Light dropout within residual blocks

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class FlexibleDecoder(nn.Module):
    def __init__(
        self,
        total_conv_layers=5,
        total_pool_layers=2,
        initial_channels=64,
        convs_per_block=None,
        channel_growth='double',
        temperature=3.0,
        use_residual=True,
        use_batchnorm=True,
        dropout_rate=0.2
    ):
        super(FlexibleDecoder, self).__init__()
        self.total_conv_layers = total_conv_layers
        self.total_pool_layers = total_pool_layers
        self.initial_channels = initial_channels
        self.channel_growth = channel_growth
        self.temperature = temperature
        self.use_residual = use_residual
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        # If convs_per_block is not specified, distribute conv layers evenly
        if convs_per_block is None:
            if total_pool_layers > 0:
                convs_per_block = math.ceil(total_conv_layers / total_pool_layers)
            else:
                convs_per_block = total_conv_layers

        self.features = self._make_layers(convs_per_block)
        
        # Add dropout before classifier for regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            MCDropout(p=self.dropout_rate),
            nn.Linear(self.final_num_channels, 128),  # Intermediate layer
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            MCDropout(p=self.dropout_rate),
            nn.Linear(128, 1),
            TemperatureScaledSigmoid(temperature=self.temperature),
        )
        
        # Apply proper weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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

            # Add convolutional layers (with/without residual structure)
            if self.use_residual and convs_in_this_block >= 2:
                # Create downsample layer if channel dimension changes
                downsample = None
                if in_channels != current_channels:
                    downsample = nn.Sequential(
                        nn.Conv2d(in_channels, current_channels, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(current_channels)
                    )
                
                # Add residual block
                layers.append(ResidualBlock(in_channels, current_channels, downsample=downsample))
                in_channels = current_channels
                
                # Account for the 2 conv layers in each residual block
                conv_layers_used = 2
                conv_layers_remaining -= conv_layers_used
                
                # Add remaining conv layers in this block
                remaining_block_convs = convs_in_this_block - conv_layers_used
                for _ in range(remaining_block_convs):
                    if remaining_block_convs <= 0:
                        break
                    
                    if self.use_batchnorm:
                        layers.append(nn.Conv2d(in_channels, current_channels, kernel_size=3, padding=1))
                        layers.append(nn.BatchNorm2d(current_channels))
                        layers.append(nn.ReLU(inplace=True))
                    else:
                        layers.append(nn.Conv2d(in_channels, current_channels, kernel_size=3, padding=1))
                        layers.append(nn.ReLU(inplace=True))
                    
                    layers.append(MCDropout(p=self.dropout_rate))
                    in_channels = current_channels
                    conv_layers_remaining -= 1
            else:
                # Traditional stack of convolutions (no residual)
                for _ in range(convs_in_this_block):
                    if self.use_batchnorm:
                        layers.append(nn.Conv2d(in_channels, current_channels, kernel_size=3, padding=1))
                        layers.append(nn.BatchNorm2d(current_channels))
                        layers.append(nn.ReLU(inplace=True))
                    else:
                        layers.append(nn.Conv2d(in_channels, current_channels, kernel_size=3, padding=1))
                        layers.append(nn.ReLU(inplace=True))
                    
                    layers.append(MCDropout(p=self.dropout_rate))
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

    def forward(self, x):
        # Apply random input jitter during training AND inference for adversarial robustness
        if random.random() < 0.5:
            x = x + torch.randn_like(x) * 0.1
            x = torch.clamp(x, 0, 1)  # Ensure pixel values remain valid
        
        x = self.features(x)
        x = self.classifier(x)
        return x
        
    def forward_with_feature_maps(self, x):
        """Return intermediate feature maps for visualization and analysis"""
        feature_maps = []
        
        # Extract features from each layer
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, (nn.ReLU, ResidualBlock)):
                feature_maps.append(x)
        
        x = self.classifier(x)
        return x, feature_maps
