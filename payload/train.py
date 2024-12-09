import os
import logging
import json
import shutil
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# Checks KAGGLE environment
# KAGGLE = True if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost') == 'Interactive' else False
KAGGLE = True if os.path.exists('/kaggle/input') else False

if not KAGGLE:
    from model import get_model

import sys
IN_COLAB = 'google.colab' in sys.modules

# Config
config = {
    'model_name': 'vit_base',
    'batch_size_per_gpu': 32,
    # 16GB GPU → try 16-32
    # 24GB GPU → try 32-48
    # 40GB GPU → try 48-64
    'patch_size': 16, # 16x16 patch size for ViT
    'num_workers' : min(os.cpu_count() * 2 if not IN_COLAB else 4 , 16) if not KAGGLE else 4,\
    'unfreeze_layers': 3,
    'img_size': 512,
    'epochs': 35,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_margin': 0.03,  # Stop if train_auc - val_auc > margin
    'early_stopping_patience': 3    # Number of epochs to wait before stopping
}

# Create initial run directory
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"runs/run_{config['model_name']}_{run_id}"  # Initial run directory without epoch count
os.makedirs(run_dir, exist_ok=True)

# Create subdirectories
log_dir = os.path.join(run_dir, "logs")
model_dir = os.path.join(run_dir, "models")
tensorboard_dir = os.path.join(run_dir, "tensorboard")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

# Save config
with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'training.log')),
        logging.StreamHandler()
    ]
)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
logging.info(f"Using device: {device}")
logging.info(f"Number of GPUs: {num_gpus}")

# Update batch size based on number of GPUs
config['total_batch_size'] = config['batch_size_per_gpu'] * max(1, num_gpus)
logging.info(f"Total batch size: {config['total_batch_size']}")

class RSNADataset(Dataset):
    def __init__(self, df, transform=None, is_training=True):
        self.df = df
        self.transform = transform
        self.is_training = is_training
        
        # Calculate class weights here
        self.pos_weight = len(df[df['cancer'] == 0]) / len(df[df['cancer'] == 1])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        label = self.df.iloc[idx]['cancer']
        
        image = Image.open(img_path).convert('L') # Only loads gray scale to 1 channel
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

    def get_pos_weight(self):
        return self.pos_weight

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predictions = (outputs >= 0.5).float()
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        
        all_outputs.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        try:
            current_auc = roc_auc_score(all_targets, all_outputs)
        except ValueError:
            current_auc = 0.0
        
        avg_loss = running_loss / (pbar.n + 1)
        current_acc = correct / total
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{current_acc:.4f}',
            'auc': f'{current_auc:.4f}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_auc = roc_auc_score(all_targets, all_outputs)
    
    return epoch_loss, epoch_acc, train_auc

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []
    
    # Initialize progress bar
    pbar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Calculate current AUC
            try:
                current_auc = roc_auc_score(all_targets, all_outputs)
            except ValueError:  # In case only one class is present in batch
                current_auc = 0.0
            
            # Update progress bar with all metrics
            avg_loss = val_loss / (pbar.n + 1)
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{current_acc:.4f}',
                'auc': f'{current_auc:.4f}'
            })
    
    val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    val_auc = roc_auc_score(all_targets, all_outputs)
    
    return val_loss, val_acc, val_auc

def update_run_dir_name(current_run_dir, epoch):
    """Update run directory name with current epoch count"""
    base_path = os.path.dirname(current_run_dir)
    run_id = os.path.basename(current_run_dir).split('_ep')[0]  # Get original run ID without epoch
    new_run_dir = os.path.join(base_path, f"{run_id}_ep{epoch}")
    
    if os.path.exists(current_run_dir) and current_run_dir != new_run_dir:
        # Create new directories before moving
        new_tensorboard_dir = os.path.join(new_run_dir, "tensorboard")
        new_log_dir = os.path.join(new_run_dir, "logs")
        new_model_dir = os.path.join(new_run_dir, "models")
        
        os.makedirs(new_run_dir, exist_ok=True)
        os.makedirs(new_tensorboard_dir, exist_ok=True)
        os.makedirs(new_log_dir, exist_ok=True)
        os.makedirs(new_model_dir, exist_ok=True)
        
        # Move contents from old directories to new ones
        old_tensorboard_dir = os.path.join(current_run_dir, "tensorboard")
        old_log_dir = os.path.join(current_run_dir, "logs")
        old_model_dir = os.path.join(current_run_dir, "models")
        
        if os.path.exists(old_tensorboard_dir):
            for item in os.listdir(old_tensorboard_dir):
                shutil.move(os.path.join(old_tensorboard_dir, item), new_tensorboard_dir)
        if os.path.exists(old_log_dir):
            for item in os.listdir(old_log_dir):
                shutil.move(os.path.join(old_log_dir, item), new_log_dir)
        if os.path.exists(old_model_dir):
            for item in os.listdir(old_model_dir):
                shutil.move(os.path.join(old_model_dir, item), new_model_dir)
                
        # Move any remaining files
        for item in os.listdir(current_run_dir):
            if item not in ["tensorboard", "logs", "models"]:
                shutil.move(os.path.join(current_run_dir, item), new_run_dir)
                
        # Remove old directory
        shutil.rmtree(current_run_dir)
        
    return new_run_dir

def main():
    global run_dir
    if not KAGGLE and not IN_COLAB:
        # Save script to run directory
        shutil.copy(__file__, os.path.join(run_dir, 'script.py'))
        shutil.copy('model.py', os.path.join(run_dir, 'model.py'))

    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Single channel normalization
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Single channel normalization
    ])

    # Load and prepare data
    logging.info("Loading and preparing data...")
    if KAGGLE:
        base_path = '/kaggle/input/rsna-'+str(config['img_size'])+'/train_images_processed_'+str(config['img_size'])
    else:
        try:
            import kagglehub
            path = kagglehub.dataset_download("mattzhou/rsna-" + str(config['img_size']))
            base_path = os.path.join(path, 'train_images_processed_'+str(config['img_size'])+'')
        except:
            base_path = '/workspace/.cache/kagglehub/datasets/mattzhou/rsna-'+str(config['img_size'])+'/versions/1/train_images_processed_'+ str(config['img_size']) + '/'

    df = pd.read_csv('train.csv')
    df['img_path'] = df.apply(lambda row: os.path.join(base_path, str(row.patient_id), f"{str(row.image_id)}.png"), axis=1)
    
    # Log class distribution
    pos_samples = len(df[df['cancer'] == 1])
    neg_samples = len(df[df['cancer'] == 0])
    logging.info(f"Class distribution - Positive: {pos_samples}, Negative: {neg_samples}")
    logging.info(f"Positive to Negative ratio: 1:{neg_samples/pos_samples:.2f}")

    train_df, val_df = train_test_split(df, test_size=config['validation_split'], stratify=df['cancer'])
    
    train_dataset = RSNADataset(train_df, transform=train_transform, is_training=True)
    val_dataset = RSNADataset(val_df, transform=val_transform, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=config['total_batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['total_batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    # Initialize model
    model = get_model(
        config['model_name'],
        img_size=config['img_size'],
        patch_size=config['patch_size'] if  'vit' in config['model_name'] else None
    )
    model.unfreeze_last_n_layers(n=config['unfreeze_layers'])

    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    logging.info("Model summary:\n")
    logging.info(summary(model, input_size=(1, config['img_size'], config['img_size'])))

    # Use weighted BCE Loss
    pos_weight = torch.tensor([train_dataset.get_pos_weight()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Remove sigmoid from model since BCEWithLogitsLoss includes it
    if isinstance(model, nn.DataParallel):
        model.module.sigmoid = nn.Identity()
    else:
        model.sigmoid = nn.Identity()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    # TensorBoard writer
    writer = None  # Initialize writer as None

    # Training loop
    best_val_loss = float('inf')
    best_val_auc = 0.0
    metrics_list = []
    
    # Early stopping variables
    early_stop_counter = 0
    best_auc_diff = float('inf')

    for epoch in range(config['epochs']):
        # Update run directory name with current epoch
        run_dir = update_run_dir_name(run_dir, epoch + 1)
        
        # Update tensorboard writer for the new directory
        if writer is not None:
            writer.close()
        tensorboard_dir = os.path.join(run_dir, "tensorboard")
        writer = SummaryWriter(tensorboard_dir)
        
        # Training phase
        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation phase
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        
        # Calculate AUC difference for early stopping
        auc_diff = train_auc - val_auc
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'loss': train_loss,
            'accuracy': train_acc,
            'auc': train_auc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_auc': val_auc,
            'auc_diff': auc_diff
        }
        metrics_list.append(metrics)
        
        # Save metrics with updated path
        pd.DataFrame(metrics_list).to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        writer.add_scalar('AUC/difference', auc_diff, epoch)
        
        # Save best models based on both validation loss and AUC
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, 'models', 'best_loss_model.pth'))
            logging.info(f"New best validation loss: {best_val_loss:.4f}")
            
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(run_dir, 'models', 'best_auc_model.pth'))
            logging.info(f"New best validation AUC: {best_val_auc:.4f}")
        
        # Early stopping check
        if auc_diff > config['early_stopping_margin']:
            early_stop_counter += 1
            logging.info(f"Early stopping counter: {early_stop_counter}/{config['early_stopping_patience']}")
            if early_stop_counter >= config['early_stopping_patience']:
                logging.info(f"Early stopping triggered! Train AUC exceeds Val AUC by {auc_diff:.4f}")
                break
        else:
            early_stop_counter = 0
        
        # Update learning rate based on validation AUC instead of loss
        scheduler.step(-val_auc)  # Negative because scheduler is in 'min' mode
        
        # Log epoch summary
        logging.info(
            f"Epoch {epoch + 1}: "
            f"loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, auc: {train_auc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}, val_auc: {val_auc:.4f}, "
            f"auc_diff: {auc_diff:.4f}"
        )

    # Close the final writer
    if writer is not None:
        writer.close()

    # Save final model with updated path
    torch.save(model.state_dict(), os.path.join(run_dir, 'models', 'final_model.pth'))
    logging.info(f"Training completed! All files saved in: {run_dir}")

if __name__ == '__main__':
    main()
