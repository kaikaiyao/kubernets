import math  # For logarithmic calculations 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.nn.utils import spectral_norm  # For Spectral Normalization
from torch.utils.data import Subset

# ----------------------------
# 1. Hyperparameters and Configuration
# ----------------------------
batch_size = 128
image_size = 32  # Set to 64, 128, 256, etc., must be power of 2 and >=16
channels_img = 3
nz = 512
ngf = 256  # Reduced from 256 for balanced capacity
ndf = 256  # Reduced from 256 for balanced capacity
num_epochs = 100
lrG = 0.0005  # Generator learning rate
lrD = 0.0005  # Reduced Discriminator learning rate
beta1 = 0.5
sample_interval = 10  # Save images every 'sample_interval' epochs
dataset_name = 'celeba'  # Options: 'celeba', 'lsun', 'oxford_iiit_pet', 'stanford_cars'
subset = True # only for 'celeba' for now
subset_size = 50000

# ----------------------------
# 2. Device Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# 3. Data Preprocessing and Loading
# ----------------------------
# Common transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(),  # Added augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.5]*channels_img, [0.5]*channels_img)
])

# Function to load the selected dataset
def load_dataset(name, transform, batch_size):
    name = name.lower()  # Ensure the dataset name is case-insensitive
    
    if name == 'celeba':
        dataset = datasets.CelebA(root='data', split='train', download=True, transform=transform)
    
    elif name == 'lsun':
        # LSUN has multiple classes. Specify one or more classes.
        # Example classes: 'bedroom_train', 'church_outdoor_train', 'kitchen_train', etc.
        # Note: torchvision.datasets.LSUN may require manual download depending on the torchvision version.
        classes = ['bedroom_train']  # You can choose other classes as needed
        dataset = datasets.LSUN(root='data', classes=classes, transform=transform)
        print("Please ensure that the LSUN dataset is downloaded and placed in the 'data' directory.")
        print("You can download LSUN from: https://www.yf.io/p/lsun")
    
    elif name == 'oxford_iiit_pet':
        dataset = datasets.OxfordIIITPet(root='data', split='trainval', download=True, transform=transform)
    
    elif name == 'stanford_cars':
        dataset = datasets.StanfordCars(root='data', split='train', download=True, transform=transform)
    
    elif name == 'cifar10':
        dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
    
    elif name == 'svhn':
        # SVHN has multiple splits: 'train', 'test', 'extra'
        # Here, we'll use the 'train' split. Modify as needed.
        dataset = datasets.SVHN(root='data', split='train', download=True, transform=transform)
    
    else:
        raise ValueError(f"Dataset '{name}' is not supported. Choose from 'celeba', 'lsun', 'oxford_iiit_pet', 'stanford_cars', 'cifar10', 'cifar100', 'svhn'.")
    
    if subset:
        dataset = Subset(dataset, range(subset_size))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloader
# Load the selected dataset
dataloader = load_dataset(dataset_name, transform, batch_size)
print(f"Loaded dataset: {dataset_name} with {len(dataloader.dataset)} samples.")

# ----------------------------
# 4. Define Generator and Discriminator
# ----------------------------

class Generator(nn.Module):
    def __init__(self, nz, ngf, channels_img, image_size):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.nz = nz
        self.ngf = ngf
        self.channels_img = channels_img

        # Ensure image_size is a power of 2 and at least 16
        assert math.log2(image_size).is_integer() and image_size >= 16, \
            "image_size must be a power of 2 and >= 16"

        # Calculate the number of upsampling layers needed
        self.n_layers = int(math.log2(image_size)) - 1  # Corrected from -2 to -1

        layers = []
        current_size = 4  # Starting from 4x4
        current_filters = ngf * 8

        # Initial layer: nz -> ngf*8
        layers += [
            nn.ConvTranspose2d(nz, current_filters, 4, 1, 0, bias=False),
            nn.BatchNorm2d(current_filters),
            nn.ReLU(True)
        ]
        print(f"Generator: Added initial layer. Output size: {current_size}x{current_size}, Filters: {current_filters}")

        # Middle layers: ngf*8 -> ngf*4 -> ngf*2 -> ...
        for _ in range(self.n_layers - 3):
            layers += [
                nn.ConvTranspose2d(current_filters, current_filters // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(current_filters // 2),
                nn.ReLU(True)
            ]
            current_filters = current_filters // 2
            current_size *= 2
            print(f"Generator: Added middle layer. Output size: {current_size}x{current_size}, Filters: {current_filters}")

        # Final layers
        layers += [
            nn.ConvTranspose2d(current_filters, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        ]
        current_size *= 2  # Assuming two final layers
        print(f"Generator: Added final layers. Output size: {current_size}x{current_size}, Filters: {channels_img}")

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, channels_img, ndf, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.ndf = ndf
        self.channels_img = channels_img

        # Ensure image_size is a power of 2 and at least 16
        assert math.log2(image_size).is_integer() and image_size >= 16, \
            "image_size must be a power of 2 and >= 16"

        # Calculate the number of downsampling layers needed
        self.n_layers = int(math.log2(image_size)) - 2  # For image_size=128, n_layers=5

        layers = []
        current_size = image_size
        current_filters = ndf

        # Initial layer: channels_img -> ndf
        layers += [
            spectral_norm(nn.Conv2d(channels_img, current_filters, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        current_size = current_size // 2
        print(f"Discriminator: Added initial layer. Output size: {current_size}x{current_size}, Filters: {current_filters}")

        # Middle layers: ndf -> ndf*2 -> ndf*4 -> ... -> ndf*8
        for _ in range(self.n_layers - 1):
            layers += [
                spectral_norm(nn.Conv2d(current_filters, current_filters * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(current_filters * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            current_filters = current_filters * 2
            current_size = current_size // 2
            print(f"Discriminator: Added middle layer. Output size: {current_size}x{current_size}, Filters: {current_filters}")

        # Final layer: ndf*8 -> 1
        layers += [
            spectral_norm(nn.Conv2d(current_filters, 1, 4, 1, 0, bias=False)),
            nn.AdaptiveAvgPool2d(1)
            # Removed Sigmoid for BCEWithLogitsLoss
        ]
        current_size = 1
        print(f"Discriminator: Added final layers. Output size: {current_size}x{current_size}, Filters: 1")

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)  # Flatten to [N]


# Initialize models
netG = Generator(nz, ngf, channels_img, image_size).to(device)
netD = Discriminator(channels_img, ndf, image_size).to(device)

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# ----------------------------
# 5. Loss Function and Optimizers
# ----------------------------
criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss

# Create labels with label smoothing
real_label = 0.8  # Reintroduced label smoothing
fake_label = 0.0

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))  # Lower learning rate for D
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))  # Maintain learning rate for G

# Learning rate schedulers
schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=50, gamma=0.1)
schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.1)

# Fixed noise for visualization
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# ----------------------------
# 6. Training Loop
# ----------------------------
G_losses = []
D_losses = []
img_list = []

if not os.path.exists('output_images'):
    os.makedirs('output_images')

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update Discriminator
        ###########################
        netD.zero_grad()
        real_images = data[0].to(device)
        b_size = real_images.size(0)
        label_real_tensor = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        label_fake_tensor = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

        # Forward pass real batch
        output_real = netD(real_images)
        # Debugging: Check the range of output_real
        min_real = output_real.min().item()
        max_real = output_real.max().item()
        print(f"Epoch {epoch+1}, Batch {i}: output_real range: {min_real:.4f}, {max_real:.4f}")
        lossD_real = criterion(output_real, label_real_tensor)
        lossD_real.backward()

        # Generate fake images
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        print(f"Epoch {epoch+1}, Batch {i}: Fake images shape: {fake_images.shape}")
        output_fake = netD(fake_images.detach())
        # Debugging: Check the range of output_fake
        min_fake = output_fake.min().item()
        max_fake = output_fake.max().item()
        print(f"Epoch {epoch+1}, Batch {i}: output_fake range: {min_fake:.4f}, {max_fake:.4f}")
        lossD_fake = criterion(output_fake, label_fake_tensor)
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake

        # Removed Gradient Clipping as Spectral Normalization is applied
        optimizerD.step()

        ############################
        # (2) Update Generator
        ###########################
        netG.zero_grad()
        label_gen_tensor = torch.full((b_size,), real_label, dtype=torch.float, device=device)  # Generator tries to fool discriminator
        output = netD(fake_images)
        # Debugging: Check the range of generator output
        min_gen = output.min().item()
        max_gen = output.max().item()
        print(f"Epoch {epoch+1}, Batch {i}: Generator output range: {min_gen:.4f}, {max_gen:.4f}")
        lossG = criterion(output, label_gen_tensor)
        lossG.backward()

        # Removed Gradient Clipping as Spectral Normalization is applied
        optimizerG.step()

        # Save Losses
        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        # Print statistics
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} \
                  Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

    # Step the schedulers
    schedulerD.step()
    schedulerG.step()

    # Save generated images
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    img_grid = vutils.make_grid(fake, padding=2, normalize=True)
    img_list.append(img_grid)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(f"Epoch {epoch+1}")
    plt.imshow(np.transpose(img_grid, (1,2,0)))
    plt.savefig(f"output_images/epoch_{epoch+1}.png")
    plt.close()

    # Optionally, save models at intervals
    if (epoch+1) % 25 == 0:
        torch.save(netG.state_dict(), f"generator_epoch_{epoch+1}.pth")
        torch.save(netD.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

# ----------------------------
# 7. Save Final Models
# ----------------------------
torch.save(netG.state_dict(), "generator_final.pth")
torch.save(netD.state_dict(), "discriminator_final.pth")

# ----------------------------
# 8. Plot Losses
# ----------------------------
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("output_images/loss_plot.png")
plt.close()

# ----------------------------
# 9. Visualize Final Results
# ----------------------------
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images at Final Epoch")
plt.imshow(np.transpose(img_list[-1], (1,2,0)))
plt.savefig("output_images/generated_final.png")
plt.close()
