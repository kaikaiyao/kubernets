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

def main():
    # Verify CUDA first
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable - check MIG configuration")
    
    # Get actual device count from environment
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    num_gpus = len(visible_devices.split(',')) if visible_devices else 0
    print(f"[Init] CUDA visible devices: {visible_devices} ({num_gpus} GPUs)")

    # Initialize distributed backend
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Validate device index
    if local_rank >= num_gpus:
        raise ValueError(f"Local rank {local_rank} exceeds available {num_gpus} MIG devices")
    
    # Set device carefully
    try:
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"[Rank {local_rank}] Successfully set device {local_rank}")
    except Exception as e:
        print(f"[Rank {local_rank}] Failed to set device: {str(e)}")
        raise

    # Model setup
    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    
    # Data setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=(local_rank == 0), transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Training setup
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

            if batch_idx % 100 == 0 and dist.get_rank() == 0:
                print(f"Epoch [{epoch}/5] Step [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
                for dev in range(world_size):
                    print(f"GPU{dev} memory: {torch.cuda.memory_allocated(dev)/1024**2:.2f} MB")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()