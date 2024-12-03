import tensorflow as tf
from tensorflow.keras import layers, models, applications

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
        
        # Create the model
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        x = base_model(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs, outputs)
        
        # Configure the model for training
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
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        base_model = applications.EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3),
            pooling='avg'
        )
        
        x = base_model(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model

class BasicResNet:
    def __init__(self, img_size=256, learning_rate=0.001):
        self.img_size = img_size
        self.learning_rate = learning_rate

    def build(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        base_model = applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3),
            pooling='avg'
        )
        base_model.trainable = False
        
        x = base_model(inputs)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
