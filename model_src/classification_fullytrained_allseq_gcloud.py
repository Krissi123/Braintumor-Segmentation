import numpy as np
import os
import mlflow
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

RSEED = 123

def train_models(batch_size=32):
    run_description = """
    Fully trained models 
     - classify each slice by tumour/tissue regions in the segmentation
    """
    dataset = 'full_data_stratified'

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

    # Set up gcloud TPUs
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)

    img_height = 240
    img_width = 240
    data_dir = os.path.join('data','UPENN-GBM','slice_classification_common_stratify','train')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        color_mode="rgba",
        seed=RSEED,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        color_mode="rgba",
        seed=RSEED,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    # Calculate class weights ofr weighting accuracy
    ds_classes = []
    for _, batch_classes in train_ds:
        ds_classes.append(batch_classes.numpy())

    ds_classes = np.concatenate(ds_classes)

    class_weight = compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(ds_classes),
        y=ds_classes
    )

    class_weight = dict(zip(np.unique(ds_classes), class_weight))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)


    with mlflow.start_run(nested=True):
        # Run small model
        with strategy.scope():
            small_model = Sequential([
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
            small_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

        small_model.summary()
        mlflow.set_tags({
            'dataset': dataset,
            'mlflow.note.content': small_model.summary(),
            })
        mlflow.log_param('ds_batch_size', batch_size)
        mlflow.log_param('ds_validation_batch_size', batch_size)
        small_model_epochs=50
        small_model_history = small_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=small_model_epochs,
        class_weight=class_weight
        )

    with mlflow.start_run(nested=True):
        # Create and fit larger model
        with strategy.scope():
            larger_model = Sequential([
            layers.Rescaling(1./(2**8-1), input_shape=(img_height,
                                        img_width,
                                        4)),
            layers.Conv2D(8, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, name="outputs")
            ])

            larger_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

        larger_model.summary()
        mlflow.set_tags({
            'dataset': dataset,
            'mlflow.note.content': larger_model.summary(),
            })
        mlflow.log_param('ds_batch_size', batch_size)
        mlflow.log_param('ds_validation_batch_size', batch_size)
        larger_model_epochs = 100
        larger_model_history = larger_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=larger_model_epochs,
        class_weight=class_weight
        )

    with mlflow.start_run(nested=True):
        # Data augmentation layer
        data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        4)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
        )
        # Apply augmentation to training data set
        aug_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        aug_ds = aug_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        # Create and fit larger model with augmented data
        with strategy.scope():
            augmented_model = Sequential([
            layers.Rescaling(1./(2**8-1), input_shape=(img_height,
                                        img_width,
                                        4)),
            layers.Conv2D(8, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, name="outputs")
            ])

            augmented_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

        mlflow.set_tags({
            'dataset': dataset + '(augmented)',
            'mlflow.note.content': augmented_model.summary(),
            })
        mlflow.log_param('ds_batch_size', batch_size)
        mlflow.log_param('ds_validation_batch_size', batch_size)
        augmented_model_epochs = 200
        augmented_model_history = augmented_model.fit(
        aug_ds,
        validation_data=val_ds,
        epochs=augmented_model_epochs,
        class_weight=class_weight
        )


if __name__ == '__main__':
    train_models()
