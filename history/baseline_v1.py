import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import kagglehub
from tqdm import tqdm

from models import ResNetModel, CustomCNN

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

batch_size = 256
num_epochs = 35


# Download dataset if not running on Kaggle
try:
    path = kagglehub.dataset_download("radek1/rsna-mammography-images-as-pngs")
    base_path = path + '/images_as_pngs_cv2_256/'
except:
    base_path = '.cache/rsna-mammography-images-as-pngs/images_as_pngs_cv2_256/'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class BreastCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if 'cancer' in self.df.columns:
            label = torch.tensor(self.df.iloc[idx]['cancer'], dtype=torch.float32)
            return image, label
        return image


def train_epoch(model, train_loader, criterion, optimizer, device, rank):
    model.train()
    running_loss = 0.0
    
    if rank == 0:
        pbar = tqdm(train_loader, desc='Training')
    else:
        pbar = train_loader
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device, rank):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in (tqdm(val_loader, desc='Validation') if rank == 0 else val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()
            
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(val_loader)
    return val_loss, predictions, true_labels

def train(rank, world_size):
    setup(rank, world_size)

    # Load and prepare data
    train_df = pd.read_csv('train.csv')
    train_df['img_path'] = f'{base_path}/train_images_processed_cv2_256/' + train_df.patient_id.astype(str) + '/' + train_df.image_id.astype(str) + '.png'
    
    # Split data
    train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['cancer'], random_state=42)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BreastCancerDataset(train_data, transform=train_transform)
    val_dataset = BreastCancerDataset(val_data, transform=val_transform)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup device and model
    device = torch.device(f'cuda:{rank}')
    # model = ResNetModel(pretrained=True).to(device)
    model = CustomCNN().to(device)

    model = DDP(model, device_ids=[rank])
    
    # Handle class imbalance
    pos_weight = torch.tensor([(len(train_df) - train_df['cancer'].sum()) / train_df['cancer'].sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, rank)
        val_loss, predictions, true_labels = validate(model, val_loader, criterion, device, rank)
        
        if rank == 0:
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), 'best_model.pth')
        
        scheduler.step(val_loss)
        
    cleanup()

if __name__ == "__main__":
    # Get the number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Found {world_size} GPUs! Training with distributed data parallel")
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("No multiple GPUs found. Training with single GPU or CPU")
        train(0, 1)
