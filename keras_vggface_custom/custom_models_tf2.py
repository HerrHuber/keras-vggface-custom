# -*- coding: utf-8 -*-
# This Program

import time
import tensorflow as tf

import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    Activation, MaxPooling2D, Add, ZeroPadding2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras import datasets, models, Input

from ResNet_keras.models import ResNet50
from use_vggface_data import load_vggface_data


# Train a ResNet50 implementation from Benedikt Huber
# loads data and trains the model in batches to avoid running out of memory
def ResNet50_classifier():
    data_dir = '../../VGG_Datasets/output/cropped_images'
    label_dir = '../../VGG_Datasets/output/labels'
    batch_size = 32
    epochs = 1
    img_height = 224
    img_width = 224
    #AUTOTUNE = tf.data.AUTOTUNE
    AUTOTUNE = 1

    train_ds, test_ds, class_names = load_vggface_data(
        data_dir,
        label_dir,
        batch_size,
        img_height,
        img_width,
        AUTOTUNE
    )

    classes = len(class_names)

    model = ResNet50(input_shape=(224, 224, 3), classes=10)

    last_layer = model.get_layer('activation_48').output

    X = AveragePooling2D(pool_size=(7, 7), name="avg_pool")(last_layer)
    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes))(X)
    new_model = models.Model(model.input, X)

    new_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = new_model.fit(
        train_ds.batch(batch_size),
        epochs=epochs,
        validation_data=test_ds
    )

    """
    # Evaluate the network
    results = new_model.predict(test_ds)
    xxx = new_model.evaluate(test_ds)
    print(xxx)
    """


# train a ResNet50 topless model from scratch
# uses tripletSemiHardLoss function to calculation image embeddings
# see: https://www.tensorflow.org/addons/tutorials/losses_triplet
def ResNet50_tripletloss():
    """
    def _normalize_img(img, label):
        img = tf.cast(img, tf.float32) / 255.
        return (img, label)

    train_dataset, test_dataset = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)
    print(train_dataset.take(1))

    # Build your input pipelines
    train_dataset = train_dataset.shuffle(1024).batch(32)
    train_dataset = train_dataset.map(_normalize_img)

    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.map(_normalize_img)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=None),  # No activation on final dense layer
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
    ])
    """

    data_dir = '../../VGG_Datasets/output/cropped_images'
    label_dir = '../../VGG_Datasets/output/labels'
    batch_size = 2
    img_height = 224
    img_width = 224
    #AUTOTUNE = tf.data.AUTOTUNE
    AUTOTUNE = 1
    train_ds, test_ds, class_names = load_vggface_data(
        data_dir,
        label_dir,
        batch_size,
        img_height,
        img_width,
        AUTOTUNE
    )

    model = ResNet50(input_shape=(224, 224, 3), classes=10)
    # add necessery layers

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss()
    )

    # Train the network
    history = model.fit(
        train_ds.batch(batch_size),
        epochs=1
    )

    """
    # Evaluate the network
    results = model.predict(test_ds)
    xxx = model.evaluate(test_ds)
    print(xxx)

    print(results[:5])
    for image, label in train_ds.take(5):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())
    """


def main():
    print(time.time())
    #ResNet50_tripletloss()
    ResNet50_classifier()


if __name__ == "__main__":
    main()
#
#
#
#
#
#
#
#
#
#
