import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        return self.fc(x)

def print_rank0(message):
    if dist.get_rank() == 0:
        print(message)

    # Verify CUDA availability FIRST
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Check MIG configuration")
    
    # Initialize distributed backend
    dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"Rank {local_rank}: CUDA device count = {torch.cuda.device_count()}")
    
    # Explicitly set device for MIG compatibility
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Print device information from all ranks
    print(f"[Rank {local_rank}/{world_size}] Using device: {torch.cuda.get_device_name(device)}")
    print(f"[Rank {local_rank}] CUDA visible devices: {torch.cuda.device_count()}")
    print(f"[Rank {local_rank}] Allocated memory: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    # Verify we're using all available GPUs
    if local_rank == 0:
        available_gpus = torch.cuda.device_count()
        assert world_size == available_gpus, \
            f"Configured {world_size} processes but found {available_gpus} GPUs!"
        print_rank0(f"\n🚀 Starting training with {world_size} GPUs:")
        print_rank0(f"Available GPUs: {available_gpus}")
        print_rank0(f"World size: {world_size}\n")

    # Create model and wrap with DDP
    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    print(f"[Rank {local_rank}] Model placed on {device}")

    # Prepare dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=(local_rank == 0), transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(5):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                # Synchronize and print from all ranks
                torch.distributed.barrier()
                if local_rank == 0:
                    print(f"\nEpoch [{epoch}/5] Step [{batch_idx}/{len(dataloader)}]")
                    print(f"Rank {local_rank} Loss: {loss.item():.4f}")
                    # Print memory usage across all devices
                    for dev in range(world_size):
                        print(f"GPU{dev} memory used: {torch.cuda.memory_allocated(dev)/1024**2:.2f} MB")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()