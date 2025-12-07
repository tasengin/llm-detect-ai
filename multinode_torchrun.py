import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
from torch import amp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model-size', type=str, default='medium', 
                       choices=['small', 'medium', 'large', 'xl'])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--checkpoint-path', type=str, default='./outputs/checkpoint.pt')
    return parser.parse_args()

def get_model(model_size):
    model_dict = {
        'small': models.resnet18,
        'medium': models.resnet50,
        'large': models.resnet101,
        'xl': models.resnet152
    }
    return model_dict[model_size]()

def setup_distributed():
    """Initialize distributed training environment"""
    # Get environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Set the GPU device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    dist.init_process_group(backend="nccl", device_id=torch.cuda.current_device())
    
    return local_rank, world_size, rank

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def main():
    args = parse_args()

    # Setup distributed training
    local_rank, world_size, rank = setup_distributed()
    
    # Print debug information from rank 0
    if rank == 0:
        print(f"Starting training with {world_size} GPUs")
        print(f"Model: {args.model_size}, Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
    
    # Create model and move to GPU
    model = get_model(args.model_size)
    model = model.cuda(local_rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Setup optimizer and scaler for mixed precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = amp.GradScaler('cuda')
    
    # Setup dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Download dataset only on rank 0
    if rank == 0:
        datasets.CIFAR10(root='./data', train=True, download=True)
    dist.barrier()  # Wait for rank 0 to download
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                    download=False, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                      rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,  # Reduced for stability
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda(local_rank, non_blocking=True)
            targets = targets.cuda(local_rank, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if batch_idx % 20 == 0 and rank == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] "
                      f"[Batch {batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # Save checkpoint from rank 0
        if args.save_checkpoint and rank == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': running_loss / len(train_loader)
            }
            torch.save(checkpoint, f"{args.checkpoint_path}.epoch{epoch+1}")
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    if rank == 0:
        print("Training completed successfully!")
    
    cleanup()

if __name__ == "__main__":
    main()

