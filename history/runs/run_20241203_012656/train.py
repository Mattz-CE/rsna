import tensorflow as tf
from tensorflow.keras import layers, models, applications
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import kagglehub
import os
import logging
from datetime import datetime
import json
import shutil
from model import BasicResNet, AdvancedResNet, EfficientNetModel

# Create run directory
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"runs/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)

# Create subdirectories
log_dir = os.path.join(run_dir, "logs")
model_dir = os.path.join(run_dir, "models")
tensorboard_dir = os.path.join(run_dir, "tensorboard")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'training.log')),
        logging.StreamHandler()
    ]
)

# Config
config = {
    'model_name': 'AdvancedResNet',  # Choose model from: BasicResNet, AdvancedResNet, EfficientNetModel
    'batch_size_per_gpu': 256,
    'img_size': 256,
    'epochs': 35,
    'learning_rate': 0.001,
    'validation_split': 0.2
}

# Save config
with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# Verify GPU setup and set strategy
physical_devices = tf.config.list_physical_devices('GPU')
logging.info(f"Num GPUs Available: {len(physical_devices)}")
strategy = tf.distribute.MirroredStrategy()
logging.info(f"Number of devices in strategy: {strategy.num_replicas_in_sync}")

# Update batch size based on number of GPUs
config['total_batch_size'] = config['batch_size_per_gpu'] * strategy.num_replicas_in_sync

def create_dataset(df, is_training=True):
    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [config['img_size'], config['img_size']])
        img = tf.cast(img, tf.float32) / 255.0
        
        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((
        df['img_path'].values, 
        df['cancer'].values
    ))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config['total_batch_size']).prefetch(tf.data.AUTOTUNE)
    return dataset

# Save scripts to run directory
shutil.copy(__file__, os.path.join(run_dir, 'train.py'))
shutil.copy('models.py', os.path.join(run_dir, 'models.py'))

# Load and prepare data
logging.info("Loading and preparing data...")
try:
    path = kagglehub.dataset_download("radek1/rsna-mammography-images-as-pngs")
    base_path = path + '/images_as_pngs_cv2_256/'
except:
    base_path = '.cache/rsna-mammography-images-as-pngs/images_as_pngs_cv2_256/'

df = pd.read_csv('train.csv')
df['img_path'] = base_path + '/train_images_processed_cv2_256/' + df.patient_id.astype(str) + '/' + df.image_id.astype(str) + '.png'
train_df, val_df = train_test_split(df, test_size=config['validation_split'], stratify=df['cancer'])

# Create datasets
train_dataset = create_dataset(train_df, True)
val_dataset = create_dataset(val_df, False)

# Build model
with strategy.scope():
    model_class = globals()[config['model_name']]
    model = model_class(
        img_size=config['img_size'],
        learning_rate=config['learning_rate']
    ).build()

# Train model
logging.info(f"Starting training with {config['model_name']}...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config['epochs'],
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            patience=3,
            factor=0.5,
            verbose=1
        )
    ]
)

# Save final model
model.save(os.path.join(model_dir, 'final_model.keras'))
logging.info(f"Training completed! All files saved in: {run_dir}")