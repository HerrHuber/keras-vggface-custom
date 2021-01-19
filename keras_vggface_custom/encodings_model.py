# -*- coding: utf-8 -*-
# This Program

import time
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

import tensorflow as tf

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filepath, model=None):
    im = Image.open(filepath)
    face = np.array(im)
    # convert into an array of samples
    sample = np.asarray([face], 'float32')
    # prepare the face for the model, e.g. center pixels
    sample = preprocess_input(sample, version=2)
    if model is None:
        # create a vggface model
        model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(sample)
    return yhat


def add_to_embeddings(file, boundingbox_path='../boxes', embeddings_path='../known_embeddings', model=None):
    # calculate encoding
    embeddings = get_embeddings(os.path.join(boundingbox_path, file), model)
    # save encoding
    np.savez(os.path.join(embeddings_path, file), embeddings)
    #print(embeddings.shape)


def add_to_embeddings_tf(file, boundingbox_path, embeddings_path, model=None):
    # np.array
    embeddings = get_embeddings(os.path.join(boundingbox_path, file), model)
    # save embedding as TFRecord
    #np.savez(os.path.join(embeddings_path, file), embeddings)
    tf.io.write_file(os.path.join(embeddings_path, file), embeddings)
    #print(embeddings.shape)


def add_new_image_to_known_embeddings(imagepath):
    known_embeddings_path = '../known_embeddings'
    # get image from path
    # get boundingbox
    # get embeddings
    # save embeddings
    add_to_embeddings()
    pass


def get_identity_from_bndbox(imagename, input_dir='../boxes', known_embeddings_dir="../known_embeddings"):
    new_embeddings = get_embeddings(os.path.join(input_dir, imagename))

    # compare to saved embeddings
    saved_embeddings_names = os.listdir(known_embeddings_dir)

    saved_embeddings = np.zeros((len(saved_embeddings_names), 2048))
    res = [None] * len(saved_embeddings_names)
    distances = np.zeros((len(saved_embeddings_names)))

    for i, e in enumerate(saved_embeddings_names):
        data = np.load(os.path.join(known_embeddings_dir, e))
        temp = data['arr_0']
        saved_embeddings[i] = temp
        distances[i] = cosine(new_embeddings, temp)
        res[i] = [saved_embeddings_names[i].split('.')[0].encode(), 1 - distances[i]]

    # run image through the whole model
    im = Image.open(os.path.join(input_dir, imagename))
    face = np.array(im)
    pixels = face.astype('float32')
    samples = np.expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50')
    yhat = model.predict(samples)
    results = decode_predictions(yhat)

    # combine results of known embeddings and model predictions
    final = results[0] + res
    final_sorted = sorted(final, key=lambda l: float(l[1]), reverse=True)
    for result in final_sorted:
        print('%s: %.3f%%' % (result[0], result[1] * 100))


# 1. embeddings model (pretrained topless model)
# new Image -> pretrained forward ResNet50 topless -> new embedding
# -> euklidian distance over old embeddings -> list of predictions
# 2. common pretrained model
# new Image -> pretrained foreward ResNet50 -> list of predictions
# => combine and sort the two lists
# first item = highest probability
def encodings_model_test():
    print(time.time())
    # get one of my images
    boundingbox_path = '../boxes/'
    embeddings_path = '../embeddings/'
    images = os.listdir(boundingbox_path)
    add_to_embeddings(images[0], boundingbox_path, embeddings_path)


    # new model
    # new take image
    # calculate encoding
    im = Image.open(boundingbox_path + images[1])
    face = np.array(im)
    #plt.imshow(face)
    #plt.show()
    new_embeddings = get_embeddings(boundingbox_path + images[1])
    #print(new_embeddings.shape)
    # compare to saved encodings
    saved_embeddings_names = os.listdir(embeddings_path)
    saved_embeddings = np.zeros((len(saved_embeddings_names), 2048))
    res = [None] * len(saved_embeddings_names)
    distances = np.zeros((len(saved_embeddings_names)))
    print("saved_embeddings.shape: ", saved_embeddings.shape)
    #saved_embeddings = np.load(embeddings_path + saved_embeddings_names[0])
    #print(saved_embeddings['arr_0'])
    for i, e in enumerate(saved_embeddings_names):
        print("embeddings_path+e: ", embeddings_path + e)
        data = np.load(embeddings_path + e)
        #print(data.files)
        temp = data['arr_0']
        #print(temp)
        saved_embeddings[i] = temp
        distances[i] = cosine(new_embeddings, temp)
        res[i] = [saved_embeddings_names[i].split('.')[0].encode(), 1 - distances[i]]

    #print(saved_embeddings)
    #print(distances)
    print("res: ", res)
    print("res[0][1]: ", res[0][1])
    print("type(res[0][1]): ", type(res[0][1]))
    #res = np.array(res)
    #print("np.array(res): ", res)


    # if nothing matches
    # foreward propagation of old model


    pixels = face.astype('float32')
    samples = np.expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50')
    yhat = model.predict(samples)
    results = decode_predictions(yhat)
    #for result in results[0]:
    #    print('%s: %.3f%%' % (result[0], result[1] * 100))
    print("results[0]: ", results[0])
    print("results[0][0][1]: ", results[0][0][1])
    print("type(results[0][0][1]): ", type(results[0][0][1]))

    final = results[0] + res
    print("final results: ", final)
    final_sorted = sorted(final, key=lambda l: float(l[1]), reverse=True)
    print("final_sorted: ", final_sorted)
    for result in final_sorted:
        print('%s: %.3f%%' % (result[0], result[1] * 100))


# take all crops
# calculate embeddings
# save embeddings as numpy array (.npz)
def get_all_embeddings(input_dir, output_dir):
    before = len(os.listdir(input_dir))
    for i, filename in enumerate(os.listdir(input_dir)):
        print('%.3f%%: %s' % ((i + 1) / before * 100, filename))
        if i % 1000 == 0:
            model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        add_to_embeddings(filename, input_dir, output_dir, model=model)

    after = len(os.listdir(output_dir))
    print('Added ' + str(before - after) + ' embeddings')


# take all crops
# calculate embeddings
# save embeddings as tensor
def get_all_embeddings_tf(input_dir, output_dir):
    before = len(os.listdir(input_dir))
    for i, filename in enumerate(os.listdir(input_dir)[:1]):
        print('%.3f%%: %s' % ((i + 1) / before * 100, filename))
        if i % 1000 == 0:
            model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        add_to_embeddings_tf(filename, input_dir, output_dir, model=model)

    after = len(os.listdir(output_dir))
    print('Added ' + str(before - after) + ' embeddings')


def test_on_a_few_images():
    imagename = 'Benedikt_Huber_16.jpg'
    print('How is on the image: ', imagename)
    get_identity_from_bndbox(imagename)
    print()
    imagename = 'Benedikt_Huber_01.jpg'
    print('How is on the image: ', imagename)
    get_identity_from_bndbox(imagename)
    print()
    # source: https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Benedict_Cumberbatch_%2848470894756%29_%28cropped%29.jpg/440px-Benedict_Cumberbatch_%2848470894756%29_%28cropped%29.jpg
    imagename = 'Benedict_Cumberbatch.jpg'
    print('How is on the image: ', imagename)
    get_identity_from_bndbox(imagename)
    print()
    # source: https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Madonna_Rebel_Heart_Tour_2015_-_Stockholm_%2823051472299%29_%28cropped%29.jpg/440px-Madonna_Rebel_Heart_Tour_2015_-_Stockholm_%2823051472299%29_%28cropped%29.jpg
    imagename = 'Madonna.jpg'
    print('How is on the image: ', imagename)
    get_identity_from_bndbox(imagename)


def main():
    #input_dir = '../../VGG_Datasets/output/cropped_images'
    #output_dir = '../../VGG_Datasets/output/image_embeddings_tf'
    #get_all_embeddings_tf(input_dir, output_dir)
    #images = os.listdir(input_dir)
    #print(images[60])
    #print(get_embeddings(os.path.join(input_dir, images[60])))
    #encodings_model_test()

    # test on a few images
    test_on_a_few_images()

    # add new identity to known embeddings
    imagename = 'Benedikt_Huber_16.jpg'
    print("Add a new identity to known idetities: Benedikt Huber")
    add_to_embeddings(imagename)

    # test on a few images
    test_on_a_few_images()


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
