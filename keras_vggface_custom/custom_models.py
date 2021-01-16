# -*- coding: utf-8 -*-
# This Program

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import tensorflow as tf
#import tensorflow_datasets as tfds
import pathlib

from keras_vggface.vggface import VGGFace
from keras.engine import Model
from keras.layers import Flatten, Dense, Input


def contiue_training():
    boundingbox_path = '../boxes/'
    # read to X, Y
    classes = 2622  # assumes 2622 classes from 0 to 2621 => new class 2622
    images = os.listdir(boundingbox_path)
    m = len(images)  # number of examples
    X = np.zeros((m, 224, 224, 3))
    for i in images:
        im = Image.open(boundingbox_path + images[0])
        image = np.array(im)
        X[0, :, :, :] = image

    X = X / 255

    Y = np.ones((m, 1)) * (classes)

    """
    (train_images, train_labels), (test_images, test_labels) = ds.cifar10.load_data()
    print("train_images.shpae: ", train_images.shape)
    #print(train_images[0, :, :, :])
    print("train_labels.shpae: ", train_labels.shape)
    print(train_labels[0, :])
    """

    num_epochs = 1
    minibatch_size = 64
    # take model
    # init params with pretrained weights
    # continue training with my data and y + my class
    model = VGGFace(X.shape[1:])
    # model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    # model.compile(optimizer="adam", loss=tf.keras.losses.sparse_categorical_crossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(x=X, y=Y, epochs=num_epochs, batch_size=minibatch_size)


    # acc_loss_train = model.evaluate(x=X_train, y=Y_train)
    # acc_loss_test = model.evaluate(x=X_test, y=Y_test)
    # print()
    # print("epoch " + str(num_epochs) + ":")
    # print("Train Accuracy = " + str(acc_loss_train[1]))
    # print("Test Accuracy = " + str(acc_loss_test[1]))

    # diff = time.time() - start
    # print("Time: ", diff)

    # preds_train = model.predict(X_train)
    # preds_test = model.predict(X_test)
    # print("preds_train[:10, 0]: ", preds_train[:10, 0])
    # preds_train = (preds_train >= 0.5) * 1
    # preds_test = (preds_test >= 0.5) * 1


def continue_train_topless_binary():
    print(time.time())
    # fine tuning

    boundingbox_path = '../boxes/'
    boundingbox_path2 = '../boxes2/'
    # read to X, Y
    #classes = 2622  # assumes 2622 classes from 0 to 2621 => new class 2622
    classes = [1, 0]
    my_images = os.listdir(boundingbox_path)
    notmy_images = os.listdir(boundingbox_path2)
    m1 = len(my_images)  # number of examples me
    m2 = len(notmy_images)  # number of examples non me
    X = np.zeros((m1 + m2, 224, 224, 3))
    for i, e in enumerate(my_images):
        im = Image.open(boundingbox_path + e)
        image = np.array(im)
        X[i, :, :, :] = image

    for i, e in enumerate(notmy_images):
        im = Image.open(boundingbox_path2 + e)
        image = np.array(im)
        X[m1 + i, :, :, :] = image

    X = X / 255

    Y1 = np.ones((m1, 1)) * (classes[0])
    Y2 = np.zeros((m2, 1)) * (classes[1])
    Y = np.concatenate((Y1, Y2), axis=0)

    print("X.shape: ", X.shape)
    print("Y.shape: ", Y.shape)

    # custom parameters
    nb_class = 2
    hidden_dim = 512
    num_epochs = 2
    minibatch_size = 2

    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = Dense(nb_class, activation='softmax', name='fc8')(x)
    custom_vgg_model = Model(vgg_model.input, out)

    start = time.time()

    #custom_vgg_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
    custom_vgg_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    custom_vgg_model.fit(x=X, y=Y, epochs=num_epochs, batch_size=minibatch_size)

    acc_loss_train = custom_vgg_model.evaluate(x=X, y=Y)
    #acc_loss_test = custom_vgg_model.evaluate(x=X_test, y=Y_test)
    print()
    print("epoch " + str(num_epochs) + ":")
    print("Train Accuracy = " + str(acc_loss_train[1]))
    #print("Test Accuracy = " + str(acc_loss_test[1]))

    diff = time.time() - start
    # print("Time: ", diff)

    preds_train = custom_vgg_model.predict(X)
    # preds_test = model.predict(X_test)
    print("preds_train[:10, 0]: ", preds_train[:10, 0])
    #preds_train = (preds_train >= 0.5) * 1
    # preds_test = (preds_test >= 0.5) * 1
    #print(preds_train)


def tf_data_test():
    # https://www.tensorflow.org/guide/data#consuming_numpy_arrays
    # get all images
    # read image
    # get bounding box points from files
    # extract image name
    # assign label => Y
    # get bounding box
    # scale to 224 x 224
    # ...
    train, test = tf.keras.datasets.fashion_mnist.load_data()

    images, labels = train
    images = images / 255.0
    labels = labels.astype(np.int32)

    fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

    """
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    """
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #model.fit(fmnist_train_ds, epochs=2)

    #loss, accuracy = model.evaluate(fmnist_train_ds)
    #print("Loss :", loss)
    #print("Accuracy :", accuracy)

    model.fit(fmnist_train_ds.repeat(), epochs=2, steps_per_epoch=20)

    loss, accuracy = model.evaluate(fmnist_train_ds.repeat(), steps=10)
    print("Loss :", loss)
    print("Accuracy :", accuracy)


def main():
    # https://www.tensorflow.org/tutorials/load_data/images#load_using_keraspreprocessing
    # downloand data
    print(tf.__version__)

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    #data_dir = tf.keras.utils.get_file(origin=dataset_url,
    #                                   fname='flower_photos',
    #                                   untar=True)
    data_dir = '../../VGG_Datasets/output/images'
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*.jpg')))
    print(image_count)
    # 43992 images in output

    examples = list(data_dir.glob('*'))
    example = Image.open(str(examples[0]))
    #example.show()

    batch_size = 32
    img_height = 180
    img_width = 180

    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    """

    print(str(data_dir / '*'))
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    #for f in list_ds.take(5):
    #    print(f.numpy())


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
