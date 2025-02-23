import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.net(x)

# Synthetic dataset
class FakeDataset(Dataset):
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return torch.randn(20), torch.randint(0, 10, (1,)).item()

def setup(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create dataset and sampler
    dataset = FakeDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(3):  # Run 3 epochs for demo
        sampler.set_epoch(epoch)
        for inputs, labels in loader:
            inputs = inputs.to(rank)
            labels = labels.to(rank)
            
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if rank == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f'Found {world_size} GPUs')
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)