import tensorflow as tf
import os
import logging
from datetime import datetime
import json
import shutil
from model import build_model, CustomCallback
from data import prepare_datasets

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
    'batch_size_per_gpu': 256,
    'img_size': 512,
    'epochs': 35,
    'learning_rate': 0.001,
    'validation_split': 0.2
}

# Save config
with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# Verify GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
logging.info(f"Num GPUs Available: {len(physical_devices)}")
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    logging.info(f"GPU devices: {physical_devices}")

# Set up multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()
logging.info(f"Number of devices in strategy: {strategy.num_replicas_in_sync}")

# Update batch size based on number of GPUs
config['total_batch_size'] = config['batch_size_per_gpu'] * strategy.num_replicas_in_sync
logging.info(f"Total batch size: {config['total_batch_size']}")

# Save script to run directory
shutil.copy(__file__, os.path.join(run_dir, 'script.py'))

if __name__ == "__main__":
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(config, res=config['img_size'])

    # Train model
    logging.info("Starting model training...")
    model = build_model(config, strategy)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        callbacks=[
            CustomCallback(run_dir),
            tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_dir, 'best_model.keras'),
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=3,
                factor=0.5,
                verbose=1
            )
        ]
    )

    # Save final model
    model.save(os.path.join(model_dir, 'final_model.keras'))
    logging.info(f"Training completed! All files saved in: {run_dir}")