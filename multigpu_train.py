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
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    
    # Each process only sees one device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    print(f"[Rank {local_rank}] Using device 0 (local to this process)")
    
    # Model setup
    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=[0])
    
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
                # Since each process only sees one device, we only check memory for device 0
                print(f"GPU{local_rank} memory: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()