import glob
import os
import sys 
import shutil
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib

import mlflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def train_model(learning_rate=0.001):
    run_description = """
    Test run using small dataset of 20 patients
     - classify each slice as healthy or tumour depending on whether it contains
       core tumour in the segmentation
    """
    dataset = 'testset-20'
    # Set information for mlflow
    mlflow_tracking_uri = os.getenv('MLFLOW_URI')
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow_expt = os.getenv('CLASSIFICATION_EXPT')
    if mlflow_expt:
        mlflow.set_experiment(mlflow_expt)
    mlflow.tensorflow.autolog(log_models=False)
    mlflow.set_tags({
            'dataset': dataset,
            'mlflow.note.content': run_description,
    })
    
    data_dir = os.path.join('data','mri-datasets','first-20-testset','dataset_multiseq')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'{data_dir} not found')

    batch_size = 32
    mlflow.log_param('ds_batch_size', batch_size)
    mlflow.log_param('ds_validation_batch_size', batch_size)
    img_height = 240
    img_width = 240

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        color_mode="rgba",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        color_mode="rgba",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    class_names = train_ds.class_names
    print(class_names)
    
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./(2**8-1))

    num_classes = len(class_names)

    model = Sequential([
    layers.Rescaling(1./(2**8-1), input_shape=(img_height, img_width, 4)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()

    epochs=10
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    )

if __name__ == '__main__':
    train_model()