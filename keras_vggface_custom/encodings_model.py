# -*- coding: utf-8 -*-
# This Program

import time
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

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


def add_to_embeddings(file, boundingbox_path, embeddings_path, model=None):
    # calculate encoding
    embeddings = get_embeddings(os.path.join(boundingbox_path, file), model)
    # save encoding
    np.savez(os.path.join(embeddings_path, file), embeddings)
    #print(embeddings.shape)


# 1. embeddings model (pretrained topless model)
# new Image -> pretrained forward topless -> new embedding
# -> euklidian distance over old embeddings -> list of predictions
# 2. common pretrained model
# new Image -> pretrained foreward -> list of predictions
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
    print(saved_embeddings.shape)
    #saved_embeddings = np.load(embeddings_path + saved_embeddings_names[0])
    #print(saved_embeddings['arr_0'])
    for i, e in enumerate(saved_embeddings_names):
        print(embeddings_path + e)
        temp = np.load(embeddings_path + e)
        #print(temp.files)
        tempp = temp['arr_0']
        #print(tempp)
        saved_embeddings[i] = tempp
        distances[i] = cosine(new_embeddings, tempp)
        res[i] = [saved_embeddings_names[i].split('.')[0].encode(), 1 - distances[i]]

    #print(saved_embeddings)
    #print(distances)
    print(res)
    res = np.array(res)
    print(res)


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
    print(results[0])

    final = np.concatenate((results, res))
    print(final)


# take all crops
# calculate embeddings
# save embeddings
def get_all_embeddings(input_dir, output_dir):
    before = len(os.listdir(input_dir))
    for i, filename in enumerate(os.listdir(input_dir)):
        print('%.3f%%: %s' % ((i + 1) / before * 100, filename))
        if i % 1000 == 0:
            model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        add_to_embeddings(filename, input_dir, output_dir, model=model)

    after = len(os.listdir(output_dir))
    print('Added ' + str(before - after) + ' embeddings')


def main():
    input_dir = '../../VGG_Datasets/output/cropped_images'
    output_dir = '../../VGG_Datasets/output/image_embeddings'
    get_all_embeddings(input_dir, output_dir)
    #images = os.listdir(input_dir)
    #print(images[60])
    #print(get_embeddings(os.path.join(input_dir, images[60])))


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
