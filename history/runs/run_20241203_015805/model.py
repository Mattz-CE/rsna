import tensorflow as tf
from tensorflow.keras import layers, models, applications

class BasicResNet:
    def __init__(self, img_size=256, learning_rate=0.001):
        self.img_size = img_size
        self.learning_rate = learning_rate

    def build(self):
        base_model = applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3),
            pooling='avg'
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model

class AdvancedResNet:
    def __init__(self, img_size=256, learning_rate=0.001):
        self.img_size = img_size
        self.learning_rate = learning_rate

    def build(self):
        base_model = applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3),
            pooling=None
        )
        # Unfreeze some layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model

class EfficientNetModel:
    def __init__(self, img_size=256, learning_rate=0.001):
        self.img_size = img_size
        self.learning_rate = learning_rate

    def build(self):
        base_model = applications.EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3)
        )
        base_model.trainable = True
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model