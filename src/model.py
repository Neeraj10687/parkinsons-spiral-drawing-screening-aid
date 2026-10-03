"""
CNN model for spiral image classification (v4 — working version).
"""
import tensorflow as tf
from tensorflow.keras import layers, models

INPUT_SHAPE = (256, 256, 1)

def create_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.028),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomFlip('horizontal'),
    ], name='data_augmentation')

def build_model(input_shape=INPUT_SHAPE, learning_rate=1e-4):
    augmentation = create_augmentation()
    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = augmentation(inputs)
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='spiral_cnn')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model

if __name__ == "__main__":
    m = build_model()
    m.summary()
