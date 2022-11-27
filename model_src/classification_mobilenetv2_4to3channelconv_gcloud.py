import numpy as np
import os
import mlflow
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

RSEED = 123

def train_models(
    batch_size=32, 
    patience=0, 
    min_delta=0.001,
    dropout_rate=0.2,
    initial_learning_rate=0.0001
):
    
    run_name_params = (
        f'bs{batch_size}'
        f'_pat{patience}'
        f'_del{min_delta}'
        f'_dr{dropout_rate}'
        f'_lr{initial_learning_rate}'
    )
    
    print("Using parameters")
    print(run_name_params)

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

    mlflow.log_param('ds_batch_size', batch_size)
    mlflow.log_param('ds_validation_batch_size', batch_size)
    

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
    
    margin = 8
    scaled_height = img_height - 2*margin
    scaled_width = img_width - 2*margin

    # Build layers for model with fixed base
    with strategy.scope():
        crop_layer = tf.keras.layers.Cropping2D(margin)
        rescale_initial = tf.keras.layers.Rescaling(1./255)
        conv_4to3_channel = tf.keras.layers.Conv2D(3,1,padding='same',activation='tanh')
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(scaled_width,scaled_height,3),
            include_top=False,
            weights='imagenet'
        )
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(num_classes)

        base_model.trainable = False
        
        inputs = tf.keras.Input(shape=(img_width, img_height, 4))
        x = crop_layer(inputs)
        x = rescale_initial(x)
        x = conv_4to3_channel(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = prediction_layer(x)
    
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            )
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate,),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        
    # Initial fit of classification and 4 to 3 channel layers
    with mlflow.start_run(
        run_name=f'fixed_{run_name_params}',
        nested=True
    ):
        fixed_base_epochs=80
        history_fixed_base = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fixed_base_epochs,
            class_weight=class_weight,
            callbacks=[earlystopping],
        )

    # Relax top layers of base model
    base_model.trainable = True
    fix_below_layer = 100
    for layer in base_model.layers[:fix_below_layer]:
        layer.trainable = False
    with strategy.scope():
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate/10.0),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
        )

    with mlflow.start_run(
        run_name=f'partial_{run_name_params}',
        nested=True
    ):
        partial_relax_epochs=history_fixed_base.epoch[-1] + 100 
        history_partial_relax = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=partial_relax_epochs,
            initial_epoch=history_fixed_base.epoch[-1]+1,
            class_weight=class_weight,
            callbacks=[earlystopping],
        )

    # Fully relax model
    model.trainable = True

    with strategy.scope():
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate/10.0),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    with mlflow.start_run(
        run_name=f'relax_{run_name_params}',
        nested=True
    ):
        full_relax_epochs=history_partial_relax.epoch[-1] + 100
        history_full_relax = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=full_relax_epochs,
            initial_epoch=history_partial_relax.epoch[-1]+1,
            class_weight=class_weight,
            callbacks=[earlystopping],
        )
  

if __name__ == '__main__':

    # Set information for mlflow
    run_description = """
    Fully trained models 
     - classify each slice by tumour/tissue regions in the segmentation
    """
    dataset = 'full_data_stratified'
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
    
    # Run training
    for patience in [0, 3, 5]:
        train_models(patience=patience)


