# -*- coding: utf-8 -*-
# This Program

import time
import os
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
import urllib
import threading
import queue

from mtcnn.mtcnn import MTCNN

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions

import tensorflow as tf

from tensorflow.keras import datasets as ds


def extract_face(filename, size=(224, 224)):
    img = plt.imread(filename)
    # img = mpimg.imread(filename)
    # imgplot = plt.imshow(img)
    # plt.show()
    detector = MTCNN()
    result = detector.detect_faces(img)

    # extract the bounding box from the first face
    x1, y1, width, height = result[0]['box']
    x2, y2 = x1 + width, y1 + height

    face = img[y1:y2, x1:x2]

    # resize face to the model size
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    return np.asarray(image)


def test_extract_face():
    filename = "../images/Benedikt_Huber_01.jpg"
    img = plt.imread(filename)
    plt.imshow(img)
    plt.show()
    face = extract_face(filename, size=(224, 224))
    plt.imshow(face)
    plt.show()


def recognise_face(face):
    # convert one face into samples
    pixels = face.astype('float32')
    samples = np.expand_dims(pixels, axis=0)
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50')
    # perform prediction
    yhat = model.predict(samples)
    # convert prediction into names
    results = decode_predictions(yhat)
    # display most likely results
    for result in results[0]:
        print('%s: %.3f%%' % (result[0], result[1] * 100))


def test_recognise_face():
    # load the photo and extract the face
    # filename = "Ben.jpg"
    # filename = "Benedict-Cumberbatch-Sherlock-.jpg"
    # filename = "Bild1.jpg"
    # filename = "Sharon_Stone_Cannes_2013_2.jpg"
    filename = "../images/Benedikt_Huber_01.jpg"
    img = plt.imread(filename)
    plt.imshow(img)
    plt.show()

    face = extract_face(filename)
    plt.imshow(face)
    plt.show()
    recognise_face(face)


def download_to_file(filename, id, url, download_path):
    name = filename.split('.')[0]
    extension = url.split('.')[-1]
    download_filename = download_path + name + '/' + id + '.' + extension
    #print(download_filename)
    if not os.path.exists(download_filename):
        if not os.path.exists(download_path + name):
            os.makedirs(download_path + name)
        try:
            urllib.request.urlretrieve(url, download_filename)
            print("Downloaded: ", url)
        except Exception as e:
            print("Try downloading: ", url, '\n', e)
    else:
        print("Already downloaded")
    return download_filename


def downloader(q):
    filename, id, url, download_path = q.get()
    #print('in downloader')
    download_to_file(filename, id, url, download_path)
    q.task_done()


def extract_bounding_box(filename, box, output_file):
    img = Image.open(filename)
    #box = (100, 100, 400, 400)
    region = img.crop(box)
    region.save(output_file)


# Use this to download vggface instead
# https://github.com/ndaidong/vgg-faces-utils
def download_test(stage=1):
    print(time.time())
    #test_extract_face()
    #test_recognise_face()

    path = '/home/user/Projects/ML/VGGFace/keras-vggface-custom/datasets/'
    source_path = path + 'vgg_face_dataset/files/'
    download_path = path + 'vgg_face_download/images/'
    bounding_path = path + 'vgg_face_bounding/images/'
    filename = 'Aamir_Khan.txt'


    if stage == 1:
        threads = 100
        q = queue.Queue()
        for i in range(threads):
            workers = threading.Thread(target=downloader, daemon=True, args=(q,)).start()

        files = os.listdir(source_path)
        for filename in files:

            print("Opening file: ", filename)
            with open(source_path + filename, 'r') as file:
                count = 0
                #while count < 1:
                while True:
                    count += 1
                    line = file.readline()
                    if not line:
                        break
                    #print("Line{}: {}".format(count, line.strip()))
                    line_elements = line.split(' ')
                    id = line_elements[0]
                    url = line_elements[1]
                    left = line_elements[2]
                    top = line_elements[3]
                    right = line_elements[4]
                    bottom = line_elements[5]
                    pose = line_elements[6]
                    detection_score = line_elements[7]
                    curation = line_elements[8]

                    # download image to filename
                    #print("Try downloading: ", url)
                    #downlaoded_file = download_to_file(filename, id, url, download_path)
                    q.put((filename, id, url, download_path))

        q.join()

    elif stage == 2:
        #downlaoded_file = '../datasets/vgg_face_download/images/Aamir_Khan/00000009.jpg'
        downlaoded_dir = download_path + filename.split('.')[0]
        downlaoded_files = os.listdir(downlaoded_dir)
        downlaoded_file = downlaoded_files[0]
        downlaoded_file_path = downlaoded_dir + '/' + downlaoded_file
        print(downlaoded_file_path)
        left = 191.01
        top = 127.67
        right = 380.02
        bottom = 316.68
        box = (left, top, right, bottom)
        output_file = bounding_path + filename.split('.')[0] + '/' + downlaoded_file
        print(output_file)
        extract_bounding_box(downlaoded_file_path, box, output_file)

    # TODO
    # use model to predict 3 different images and my images
    # take my pictures
    # convert to dataset format/tensor
    # add 1 class to y
    # train on pretrained weights with my X and Y
    # use my model to predict 3 different images and my images

    # create function that adds new images to X and Y
    # and trains the model automaticaly


def old_save_boundingbox():
    #path = '../datasets/vgg_face_download/images/Aamir_khan/'
    path = '../datasets/vgg_face_download/images/Elon_Musk/'
    #path = '../images/'
    #boundingbox_path = '../boxes/'
    boundingbox_path = '../boxes2/'
    images = os.listdir(path)
    for image in images:
        print("Try extracting face from: ", path + image)
        if not os.path.exists(boundingbox_path + image):
            try:
                face = extract_face(path + image)
                im = Image.fromarray(face)
                im.save(boundingbox_path + image)
                # plt.imshow(face)
                # plt.show()
            except Exception as e:
                print(e)
        else:
            print(image + ' already extrated')


def main():
    pass


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
