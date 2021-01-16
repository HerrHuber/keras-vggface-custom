# -*- coding: utf-8 -*-
# This Program

import time
import pathlib
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from xml.dom import minidom
import tensorflow as tf
from tensorflow.keras import layers


class_names = None
img_height = None
img_width = None


def get_label(file_path, class_names):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    tf.print(parts)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    print(one_hot)
    # Integer encode the label
    return tf.argmax(one_hot)


#def get_label2(file_path, class_names):
def get_label2(file_path):
    global class_names

    # convert the path to a list of path components
    #parts = tf.strings.split(file_path, os.path.sep)
    #tf.print(parts)
    #print("file_path", file_path.numpy())
    print("file_path", str(file_path))
    print(file_path.shape)
    temp = file_path.decode('utf-8').split('/')[-1]
    #print(temp)
    label = get_label_from_name(temp)
    #print(label)
    # The second to last is the class-directory
    print(class_names)
    one_hot = label == class_names
    #print(one_hot)
    # Integer encode the label
    return tf.argmax(one_hot)


def get_label_tensor(file_path):
    global class_names

    filename = tf.strings.split(file_path, sep='/')[-1]
    label = tf.strings.regex_replace(filename, '_[0-9]+\.jpg', '')
    one_hot = label == class_names
    return tf.argmax(one_hot)


def decode_img(img, img_height, img_width):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


#def crop_and_decode_img(filepath, img_height, img_width):
def crop_and_decode_img(filepath):
    global img_height
    global img_width

    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)

    label_file = filepath.decode('utf-8').replace('images', 'labels').replace('jpg', 'xml')
    #label_file = tf.strings.regex_replace(filepath, 'images', 'labels')\
    #label_file = tf.strings.regex_replace(label_file, 'jpg', 'xml')
    #print(label_file)
    mydoc = minidom.parse(label_file)
    xmin = int(mydoc.getElementsByTagName('xmin')[0].firstChild.data)
    xmin = abs(xmin)
    ymin = int(mydoc.getElementsByTagName('ymin')[0].firstChild.data)
    ymin = abs(ymin)
    xmax = int(mydoc.getElementsByTagName('xmax')[0].firstChild.data)
    xmax = abs(xmax)
    ymax = int(mydoc.getElementsByTagName('ymax')[0].firstChild.data)
    ymax = abs(ymax)
    target_height = ymax - ymin
    target_width = xmax - xmin
    img = tf.image.crop_to_bounding_box(img, ymin, xmin, target_height, target_width)
    print(img_height)
    print(img_width)
    # resize the image to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    #image = img.numpy()
    #image = img.numpy().astype(np.uint8)
    #plt.imshow(image)
    #plt.show()

    return img


def crop_and_decode_img_tensor(filepath):
    global img_height
    global img_width

    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)

    """
    filename = tf.strings.regex_replace(filepath, 'images', 'labels')
    label_file = tf.strings.regex_replace(filename, 'jpg', 'xml')
    print("label_file: ", label_file)
    print("label_file.numpy(): ", label_file.numpy())
    mydoc = minidom.parse(label_file)
    xmin = int(mydoc.getElementsByTagName('xmin')[0].firstChild.data)
    xmin = abs(xmin)
    ymin = int(mydoc.getElementsByTagName('ymin')[0].firstChild.data)
    ymin = abs(ymin)
    xmax = int(mydoc.getElementsByTagName('xmax')[0].firstChild.data)
    xmax = abs(xmax)
    ymax = int(mydoc.getElementsByTagName('ymax')[0].firstChild.data)
    ymax = abs(ymax)
    target_height = ymax - ymin
    target_width = xmax - xmin
    img = tf.image.crop_to_bounding_box(img, ymin, xmin, target_height, target_width)
    print(img_height)
    print(img_width)
    """
    # resize the image to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    #image = img.numpy()
    #image = img.numpy().astype(np.uint8)
    #plt.imshow(image)
    #plt.show()

    return img


#def process_path(file_path, class_names, img_height, img_width):
def process_path(file_path):
    global class_names
    global img_height
    global img_width

    print("in process_path file_path: ", file_path)
    label = get_label2(file_path)
    img = crop_and_decode_img(file_path)
    return img, label


def process_path_tensor(file_path):
    global class_names
    global img_height
    global img_width

    print("in process_path file_path: ", file_path)
    label = get_label_tensor(file_path)
    img = crop_and_decode_img_tensor(file_path)
    return img, label


def get_label_from_name(name):
    # a.replace(a.split('_')[-1], '')[:-1]
    # conevert 'a_j__buckley_00000045.jpg'
    # to 'a_j__buckley'
    return name.replace(name.split('_')[-1], '')[:-1]


def delete_currupt(input_dir, output_dir):
    before = len(os.listdir(input_dir))
    for filename in os.listdir(input_dir):
        try:
            im = Image.open(input_dir + "/" + filename)
            exif_data = im._getexif()
            #with Image.open(input_dir + "/" + filename) as im:
            #    #print('ok')
            #    pass
        except Exception as e:
            print('Moving corrupt images to ' + output_dir)
            print(input_dir + "/" + filename)
            print(e)
            #os.remove(input_dir + "/" + filename)
            os.rename(input_dir + "/" + filename, output_dir + "/" + filename)

    after = len(os.listdir(input_dir))
    print('Moved ' + str(before - after) + ' images')


def crop_images(input_dir, label_dir, output_dir):
    global img_height
    global img_width

    before = len(os.listdir(input_dir))
    for filename in os.listdir(input_dir):
        label_file = os.path.join(label_dir, filename).replace('jpg', 'xml')
        #print(label_file)
        mydoc = minidom.parse(label_file)
        xmin = int(mydoc.getElementsByTagName('xmin')[0].firstChild.data)
        xmin = abs(xmin)
        ymin = int(mydoc.getElementsByTagName('ymin')[0].firstChild.data)
        ymin = abs(ymin)
        xmax = int(mydoc.getElementsByTagName('xmax')[0].firstChild.data)
        xmax = abs(xmax)
        ymax = int(mydoc.getElementsByTagName('ymax')[0].firstChild.data)
        ymax = abs(ymax)
        box = (xmin, ymin, xmax, ymax)
        img = Image.open(os.path.join(input_dir, filename))
        #print("Image format: ", img.format)
        #print("Image mode: ", img.mode)
        if not img.format == 'JPEG' or not img.mode == 'RGB':
            img = img.convert('RGB')
        #print("Image format: ", img.format)
        #print("Image mode: ", img.mode)
        # box = (100, 100, 400, 400)
        # box = (left, top, right, bottom)
        region = img.crop(box)
        #print(img_height)
        #print(img_width)
        # resize the image to the desired size
        region = region.resize((img_height, img_width))
        region.save(os.path.join(output_dir, filename))
        #plt.imshow(region)
        #plt.show()

    after = len(os.listdir(output_dir))
    print("Copped ", after, " images")


def tf_example():
    # https://www.tensorflow.org/tutorials/load_data/images#load_using_keraspreprocessing
    # downloand data
    print("tf.__version__: ", tf.__version__)

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                       fname='flower_photos',
                                       untar=True)

    print("data_dir: ", data_dir)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    batch_size = 32
    img_height = 180
    img_width = 180

    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    for f in list_ds.take(5):
        print(f.numpy())

    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
    print("class_names: ", class_names)

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())

    def get_label_tf(file_path):
        print("file_path: ", file_path)
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        print("parts: ", parts)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        print("one_hot: ", one_hot)
        # Integer encode the label
        temp = tf.argmax(one_hot)
        print("argmax: ", temp)
        return temp

    def decode_img_tf(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    def process_path_tf(file_path):
        label = get_label_tf(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img_tf(img)
        return img, label

    AUTOTUNE = 1
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path_tf, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path_tf, num_parallel_calls=AUTOTUNE)

    for image, label in train_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())


# create tf.data.Dataset from cropped images and labels
# feed dataset to model in batches (because dataset > memory)
def my_example():
    global class_names
    global img_height
    global img_width

    print('\n\n\n')
    print(time.time())
    # https://www.tensorflow.org/tutorials/load_data/images#load_using_keraspreprocessing
    # downloand data
    print("tf.__version__: ", tf.__version__)

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    #data_dir = tf.keras.utils.get_file(origin=dataset_url,
    #                                   fname='flower_photos',
    #                                   untar=True)
    #data_dir = '../../VGG_Datasets/output/images'
    data_dir = '../../VGG_Datasets/output/cropped_images'
    label_dir = '../../VGG_Datasets/output/labels'
    data_dir = pathlib.Path(data_dir)
    label_dir = pathlib.Path(label_dir)

    image_count = len(list(data_dir.glob('*.jpg')))
    print("image_count: ", image_count)
    # 43992 images in output

    examples = list(data_dir.glob('*'))
    example = Image.open(str(examples[0]))
    #example.show()

    batch_size = 2
    img_height = 224
    img_width = 224
    #AUTOTUNE = tf.data.AUTOTUNE
    AUTOTUNE = 1

    print("str(data_dir / *: ", str(data_dir / '*'))
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    filepath = ''
    for f in list_ds.take(5):
        filepath = f.numpy()
        print("filepath: ", filepath)

    class_names = np.array(sorted(list(set([get_label_from_name(item.name) for item in data_dir.glob('*')]))))
    print("class_names[:10]: ", class_names[:10])
    print("len(class_names): ", len(class_names))

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    print("train_ds: ", train_ds)
    val_ds = list_ds.take(val_size)

    m_train = tf.data.experimental.cardinality(train_ds).numpy()
    m_val = tf.data.experimental.cardinality(val_ds).numpy()
    print("m_train: ", m_train)
    print("m_val: ", m_val)
    print("m_train + m_val: ", m_train + m_val)

    """
    # Test get_label2()
    print()
    for f in list_ds.take(5):
        filepath = f.numpy()
        print("filepath: ", filepath)
        #tf.print(get_label2(filepath, class_names))
        tf.print(get_label2(filepath))
        #tf.print(tensor, output_stream=sys.stderr)

    # Test crop_and_decode
    print()
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    label_file = filepath.decode('utf-8').replace('images', 'labels').replace('jpg', 'xml')
    print("label_file: ", label_file)
    #bndbox = crop_and_decode_img(filepath, img_height, img_width)
    #bndbox = crop_and_decode_img(filepath)

    # Test process path
    #bndbox, label = process_path(filepath, class_names, img_height, img_width)
    bndbox, label = process_path(filepath)
    # image = img.numpy().astype(np.uint8)
    #x = bndbox.numpy().astype(np.uint8)
    #plt.imshow(x)
    #plt.show()
    #print(label)
    """

    # what is train_ds?
    for i in train_ds.take(1):
        print("i: ", i)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path_tensor, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path_tensor, num_parallel_calls=AUTOTUNE)

    for image, label in train_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    """
    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=batch_size)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    print("after configure for performance")
    """

    num_classes = len(class_names)

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(
        train_ds.batch(batch_size),
        validation_data=val_ds.batch(batch_size),
        epochs=1,
    )

    """
    model.fit(
        train_ds.repeat(),
        validation_data=val_ds,
        epochs=1,
        steps_per_epoch=20
    )

    model.train_on_batch(
        train_ds,
    )

    """

def main():
    global img_height
    global img_width

    img_height = 224
    img_width = 224
    #tf_example()
    #my_example()
    #input_dir = '../../VGG_Datasets/output/images'
    #output_dir = '../../VGG_Datasets/output/corrupt_images'
    #delete_currupt(input_dir, output_dir)
    input_dir = '../../VGG_Datasets/output/images'
    label_dir = '../../VGG_Datasets/output/labels'
    output_dir = '../../VGG_Datasets/output/cropped_images'
    crop_images(input_dir, label_dir, output_dir)  # 43956 images after delete


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
