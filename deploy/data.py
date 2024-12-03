import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os

def load_dataset(config, res):
    """Load and prepare the mammography dataset."""
    logging.info("Loading and preparing data...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("mattzhou/rsna-" + str(res))
        base_path = os.path.join(path, 'train_images_processed_'+str(res)+'')
    except:
        base_path = '/workspace/.cache/kagglehub/datasets/mattzhou/rsna-512/versions/1/train_images_processed_'+ str(res) + '/'

    df = pd.read_csv('train.csv')
    df['img_path'] = df.apply(lambda row: os.path.join(base_path, 
                                                    str(row.patient_id), 
                                                    f"{str(row.image_id)}.png"), 
                            axis=1)
    
    # Split data
    train_df, val_df = train_test_split(
        df, 
        test_size=config['validation_split'], 
        stratify=df['cancer']
    )
    
    logging.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    return train_df, val_df

def create_dataset(df, config, is_training=True):
    """Create a TensorFlow dataset from a DataFrame."""
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

def prepare_datasets(config, res):
    """Prepare training and validation datasets."""
    train_df, val_df = load_dataset(config, res=res)
    train_dataset = create_dataset(train_df, config, is_training=True)
    val_dataset = create_dataset(val_df, config, is_training=False)
    return train_dataset, val_dataset