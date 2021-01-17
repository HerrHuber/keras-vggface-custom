keras_vggface_custom
====================
custom models on vggface dataset

execute with docker container
-----------------------------
docker build -t tf2-4 .

CPU:
docker run -it --rm  -u $(id -u):$(id -g) -v /home/user/Projects/ML/VGGFace/keras-vggface-custom:/tmp/src -v /home/user/Projects/ML/VGGFace/VGG_Datasets:/tmp/datasets -w /tmp tensorflow/tensorflow:latest /bin/bash
docker run -it --rm  -u $(id -u):$(id -g) -v /home/user/Projects/ML/VGGFace/keras-vggface-custom:/home/ben/src -v /home/user/Projects/ML/VGGFace/VGG_Datasets:/home/ben/datasets -w /home/ben tf2-3 /bin/bash
Server:
docker run -it --rm  -u $(id -u):$(id -g) -v /home/ben/MLProjects/keras-vggface-custom:/home/ben/src -v /mnt/VMs/datasets/datasets:/home/ben/datasets -w /home/ben tf2-4 /bin/bash

GPU:
docker run --gpus all -it --rm  -u $(id -u):$(id -g) -v /home/user/Projects/ML/L_layer_model_tf:/tmp -w /tmp tensorflow/tensorflow:2.3-gpu-py3 /bin/bash