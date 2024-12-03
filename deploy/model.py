import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os
import logging
import pandas as pd

def build_model(config, strategy):
    with strategy.scope():
        base_model = applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(config['img_size'], config['img_size'], 3),
            pooling='avg'
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    return model

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, run_dir):
        super().__init__()
        self.metrics_file = os.path.join(run_dir, 'metrics.csv')
        self.metrics_list = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epoch'] = epoch + 1
        self.metrics_list.append(logs)
        
        # Save metrics after each epoch
        pd.DataFrame(self.metrics_list).to_csv(self.metrics_file, index=False)
        
        logging.info(
            f"Epoch {epoch + 1}: loss: {logs['loss']:.4f}, "
            f"accuracy: {logs['accuracy']:.4f}, "
            f"val_loss: {logs['val_loss']:.4f}, "
            f"val_accuracy: {logs['val_accuracy']:.4f}, "
            f"auc: {logs['auc']:.4f}, "
            f"val_auc: {logs['val_auc']:.4f}"
        )
