keras_vggface_custom
====================
Custom models using the vggface dataset 
  
This project consists of 3 working parts:
1. An encodings/embeddings model
2. A ResNet50 classifier
3. A VGG16 binary classifier 

And 2 not yet working parts:
1. An encodings/embeddings classifier
2. A ResNet50 using tripletloss



Encodings model
---------------
Encodings model uses a pretrained ResNet50 to identify a person on an image  
It uses the pretrained model from https://github.com/rcmalli/keras-vggface

To simplify adding new identities to the model without any retraining  
the model consists of two parts:  
- The embeddings part, which compares new images against known identities
  by calculating the euclidean distance between the embedding of the new  
  image and the embeddings stored in a database (directory
  containing .npz files)
- The second part, which uses a pretrained ResNet50 to predict the  
  identity of the input image
  
The model combines the predictions of the first and second part  
and sorts the combined results by probability
=> if the first element has a large probability (e.g. > 0.5)
   the identity is most likely the first element

To add a new identity to the model you have to:  
1. Get a good image of the identity and save it e.g. to images/  
2. Extract and save the face/bounding box of the new image  
   (may not work for windows paths)
   see the main function in keras_vggface_custom.py for example code
   this function uses the MTCNN python module which requires OpenCV>=4.1 and Keras>=2.0.0
   this file works with tensorflow 1.14
3. Calculate and save the embeddings of the boundingbox
   to the known embeddings file
   see the main function in encodings_model.py for example code
   this file works with tensorflow 1.14

To get predictions of identities on an image
see the main function of encodings_model.py for example code


ResNet50 classifier
-------------------
This part uses a ResNet50 implementation of Benedikt Huber  
The data is loaded and trained on in batches (if dataset > memory)  
Tested on Tensorflow 2.3  

To train this model on the vggface dataset (or a subset)  
First Download the dataset using https://github.com/ndaidong/vgg-faces-utils  
(I stopped the download after 43000 images, about 2.3 GB)  
Clean the dataset:  
- Some images are corrupt  
  `delete_corrupt()` in `use_vggface_data.py` takes care of corrupt images  
- Some images are in png format but are saved with an .jpg extension  
- Some images are gray scale  

Next crop all the images according to the corresponding bounding box  
  `crop_images()` in `use_vggface_data.py` takes care of wrong formats and the cropping  
Go to the `ResNet50_classifier()` function in `custom_models_tf2.py`  
Change the `data_dir` path, `batch_size`, `epochs`, ..., then execute the file  


VGG16 binary classifier
-----------------------
Uses a topless VGG16 pretrained on the vggface dataset
from https://github.com/rcmalli/keras-vggface
Tested on Tensorflow 1.14

This model trains a binary classifier on top of the pretrained VGG16
to classify my images and not my images

To use this classifier first download some images
then extract the bounding boxes


Encodings classifier
--------------------
Does not work yet  
The idea was to make a database consisting of embeddings of all  
images that should be identified and build a classifier that   
classifies all the embeddings  
The embeddings are calculated using a pretrained topless ResNet50  
Advantages would be:  
easy to add/delete classes  
less storage capacity (embeddings < image)  


ResNet50 using tripletloss
--------------------------
Does not work yet  
A better version of the ResNet50 classifier



Use a docker container
----------------------
Build the container image (see Dockerfile)  
TODO: install ResNet_keras from source  
`docker build -t tf2-4 .`  

CPU:  
`docker run -it --rm  -u $(id -u):$(id -g) -v /home/user/Projects/ML/VGGFace/keras-vggface-custom:/home/ben/src -v /home/user/Projects/ML/VGGFace/VGG_Datasets:/home/ben/datasets -w /home/ben tf2-4 /bin/bash`
or:  
`docker run -it --rm  -u $(id -u):$(id -g) -v /home/ben/MLProjects/keras-vggface-custom:/home/ben/src -v /mnt/VMs/datasets/datasets:/home/ben/datasets -w /home/ben tf2-4 /bin/bash`


GPU:
Follow the instructions on https://www.tensorflow.org/install/docker  
Change Dockerfile to use tensorflow/tensorflow:2.3-gpu-py3 instead of tensorflow/tensorflow:latest  
`docker build -t tf2-3-gpu .`  
`docker run --gpus all -it --rm  -u $(id -u):$(id -g) -v /home/ben/MLProjects/keras-vggface-custom:/home/ben/src -v /mnt/VMs/datasets/datasets:/home/ben/datasets -w /home/ben tf2-3-gpu /bin/bash`

