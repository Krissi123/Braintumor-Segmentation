import glob
import os
import shutil

from datetime import datetime
import numpy as np

import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import mlflow

from modules.scandata import MriScan, MriSlice, TumourSegmentation, ScanType, ScanPlane

RSEED = 123


def scaler_0_1(x):
    return x/255.0

def scaler_neg1_1(x):
    return x/127.5 - 1

def alter_segmap(x):
    return tf.where(x==4,tf.constant(3,dtype='uint8'),x)

def read_image_map(image, seg_map):
    image = tf.io.read_file(image)
    image = tf.io.decode_png(image, channels=4)
    seg_map = tf.io.read_file(seg_map)
    seg_map = tf.io.decode_png(seg_map, channels=1)
    # Change scaler below to scaler_0_1 to get initial values between 0 and 1
    return scaler_neg1_1(tf.cast(image, 'float32')) ,seg_map



def train_unet(
    batch_size=64, 
    patience=0, 
    min_delta=0.001,
    dropout_rate=0.0,
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
    run_name = f'unet_{run_name_params}_scratch'

    start_time = datetime.now().strftime('-%Y-%m-%d-%T')

    tf.config.list_logical_devices('TPU')


    try:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
    except:
        strategy = tf.distribute.get_strategy()


    # Set information for mlflow
    run_description = """
    Fully trained models 
     - classify each slice by tumour/tissue regions in the segmentation
     - Builds U-Net from scratch
    """
    dataset = 'full_data_stratified_healthytissue_dropbg'
    mlflow_tracking_uri = os.getenv('MLFLOW_URI')
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow_expt = os.getenv('SEGMENTATION_EXPT')
    if mlflow_expt:
        mlflow.set_experiment(mlflow_expt)    
    
    print(f'Logging to \n URI:{mlflow_tracking_uri}\n Expt:{mlflow_expt}')

    with mlflow.start_run(
        run_name=run_name,
        tags={
            'dataset': dataset,
        },
        description=run_description,
    ):

        buffer_size = 1000
        img_height = 240
        img_width = 240
        scan_channels = 4
        output_classes = 5

        train_image_dir = os.path.join(
            'data',
            'UPENN-GBM',
            'slice_segmentation_stratify_healthy_dropbg',
            'train',
            'image_data'
        )
        train_map_dir = os.path.join(
            'data',
            'UPENN-GBM',
            'slice_segmentation_stratify_healthy_dropbg',
            'train',
            'map_data'
        )

        # Count pixels for sample weight
        pixel_counts = [0 for x in range(output_classes)]

        for map_file in os.listdir(train_map_dir):

            seg_map = tf.io.read_file(os.path.join(train_map_dir,map_file))
            seg_map = tf.io.decode_png(seg_map, channels=1)
            
            indices,counts = np.unique(seg_map,return_counts=True)
            for i, index in enumerate(indices):
                pixel_counts[index] += counts[i]

        print(f'Number of pixels of each class: {pixel_counts}')


        image_filenames = os.listdir(train_image_dir)
        map_filenames = [filename.replace('allseq', 'map') for filename in image_filenames]
        image_filepaths = [os.path.join(train_image_dir,filename) for filename in image_filenames]
        map_filepaths = [os.path.join(train_map_dir,filename) for filename in map_filenames]


        train_image_filepaths, val_image_filepaths, train_map_filepaths, val_map_filepaths = train_test_split(
            image_filepaths, 
            map_filepaths, 
            test_size=0.2,
            random_state=RSEED,
            )
    
        train_image_data = tf.data.Dataset.list_files(train_image_filepaths, shuffle=False)
        train_map_data = tf.data.Dataset.list_files(train_map_filepaths, shuffle=False)
        train_data = tf.data.Dataset.zip((train_image_data, train_map_data))
        val_image_data = tf.data.Dataset.list_files(val_image_filepaths, shuffle=False)
        val_map_data = tf.data.Dataset.list_files(val_map_filepaths, shuffle=False)
        val_data = tf.data.Dataset.zip((val_image_data, val_map_data))

        train_batch = (
            train_data.cache()
            .shuffle(buffer_size, seed=tf.constant(RSEED,dtype='int64'))
            .repeat()
            .map(read_image_map)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        val_batch = (
            val_data
            .shuffle(buffer_size, seed=tf.constant(RSEED,dtype='int64'))
            .map(read_image_map)
            .batch(batch_size),
        )

        # Calculate class weights
        weights = 1.0/np.array(pixel_counts)
        weights = weights/np.sum(weights)

        def add_sample_weights(image, label):
            # Create an image of `sample_weights` by using the label at each pixel as an 
            # index into the `class weights` .
            sample_weights = tf.gather(weights, indices=tf.cast(label, tf.int32))

            return image, label, sample_weights

        initializer = tf.random_normal_initializer(0., 0.02)

        def horizontal_convolution(input, num_filters, activation='relu', dropout_rate=0.0):
            
            x = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=3,
                padding= 'same',
                strides=1,
                kernel_initializer=initializer,
                use_bias=False,
            )(input)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
            x = tf.keras.layers.Activation(activation=activation)(x) 
            return x


        def down_step(input, num_filters, dropout_rate=0.0):
            x = horizontal_convolution(
                input=input,
                num_filters=num_filters,
                dropout_rate=dropout_rate,
            )
            horizontal_out = horizontal_convolution(
                x, 
                num_filters=num_filters,
                dropout_rate=dropout_rate,
            )
            down_out = tf.keras.layers.MaxPooling2D(
                pool_size=2,
                strides=2, 
                padding = 'same'
            )(horizontal_out)
            return down_out, horizontal_out


        def up_step(
            up_input, 
            horizontal_input, 
            num_filters, 
            dropout_rate=0.0, 
        ):
            x = tf.keras.layers.Conv2DTranspose(
                filters=num_filters,
                kernel_size=3, 
                strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False,
            )(up_input)
            x = tf.keras.layers.Concatenate()([x, horizontal_input])
            x = horizontal_convolution(
                x, 
                num_filters=num_filters, 
                dropout_rate=dropout_rate
            )
            x = horizontal_convolution(
                x, 
                num_filters=num_filters, 
                dropout_rate=dropout_rate,
                activation='softmax',
            )
            return x

        # Define U-Net topology
        def unet(
            input_shape, 
            output_channels, 
            scale_filters=1.0, 
            dropout_rate=0.0,
            final_dropout=True
            ):

            final_dropout_rate=0.0
            if final_dropout:
                final_dropout_rate = dropout_rate

            inputs = tf.keras.layers.Input(shape=input_shape)

            down_1, cross_1 = down_step(
                inputs, 
                int(64*scale_filters), 
                dropout_rate=dropout_rate
            )  # 240 -> 120
            down_2, cross_2 = down_step(
                down_1, 
                int(128*scale_filters), 
                dropout_rate=dropout_rate
            )  # 120 -> 60
            down_3, cross_3 = down_step(
                down_2, 
                int(256*scale_filters), 
                dropout_rate=dropout_rate
            )  # 60 -> 30
            down_4, cross_4 = down_step(
                down_3, 
                int(512*scale_filters), 
                dropout_rate=dropout_rate
            )  # 30 -> 15

            bottom = horizontal_convolution(
                down_4, 
                int(1024*scale_filters), 
                dropout_rate=dropout_rate
            )

            up_4 = up_step(
                bottom, 
                cross_4, 
                int(512*scale_filters), 
                dropout_rate=dropout_rate
            )  # 15 -> 30
            up_3 = up_step(
                up_4, 
                cross_3, 
                int(256*scale_filters), 
                dropout_rate=dropout_rate
            )  # 30 -> 60
            up_2 = up_step(
                up_3, 
                cross_2, 
                int(128*scale_filters), 
                dropout_rate=dropout_rate
            )  # 60 -> 120
            up_1 = up_step(
                up_2, 
                cross_1, 
                int(64*scale_filters), 
                dropout_rate=final_dropout_rate
            )  # 120 -> 240
            
            outputs = horizontal_convolution(up_1, output_channels)

            return tf.keras.Model(inputs=inputs, outputs=outputs)

        with strategy.scope():
            input_shape = (img_width,img_height,scan_channels)
            model = unet(
                input_shape=input_shape, 
                output_channels=output_classes, 
                dropout_rate=dropout_rate
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

        earlystopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    )

        print(len(train_image_filepaths),len(val_image_filepaths))

        TRAIN_LENGTH=len(train_image_filepaths)
        EPOCHS = 60
        VAL_SUBSPLITS = 1
        VALIDATION_STEPS = len(val_image_filepaths)//batch_size//VAL_SUBSPLITS
        STEPS_PER_EPOCH = TRAIN_LENGTH // batch_size

        if not os.path.exists('model_checkpoints'):
            os.mkdir('model_checkpoints')

        checkpoint_path = os.path.join(
            'model_checkpoints',
            run_name + start_time + "-{epoch:03d}-{val_loss:.4f}.ckpt"
        )
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=False,
            save_freq='epoch',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        ) 

        model_history = model.fit(
            train_batch.map(add_sample_weights), 
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            validation_data=val_batch,
            callbacks=[earlystopping, ckpt_callback],
        )

        if not os.path.exists('models'):
            os.mkdir('models')

        
        model_file_name = os.path.join('models', run_name + start_time)
        model.save(model_file_name)
        
        
        
if __name__ == '__main__':
    for patience in 0,3,5:
        for learning_rate in 0.001,0.0005,0.00001:
            for dropout_rate in 0.0,0.2:
                print("\t Run: ",patience,learning_rate,dropout_rate)
                train_unet(
                    batch_size = 32,
                    patience=patience,
                    initial_learning_rate=learning_rate,
                    dropout_rate=dropout_rate,
                )