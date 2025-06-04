---
layout: post
title: Image Classification- Cassava Leaf Disease
date: 2021-02-16 00:00:00 +0300
description: using Deep Neural Networks

img: cassavaCOLLAGE.jpg 
tags: [Deep Learning, Neural Networks, Machine Learning]
---
# Image Classification- Cassava Leaf Disease 

### Comparison of different neural network models using PyTorch

Image classification is a supervised learning problem: define a set of target classes (objects to identify in images), and train a model to recognize them using labelled example photos. [[Source]](https://developers.google.com/machine-learning/practica/image-classification)

Deep learning, a subset of machine learning algorithms, is good at recognising patterns. Hence, it is widely used for image classification. The adjective "deep" in deep learning refers to the use of multiple layers in the network, where each layer progressively extracts higher-level features from the raw input. Deep learning models can have different architectures.

This is a project to classify the images of cassava plant leaves into five categories based on the disease affecting them. The dataset consist of 21,367 labeled images of cassava plant leaves, obtained from a [Cassava Leaf Disease Classification competition in Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview). The project aims to classify the images using the following neural network architectures and compare their performances:


1.   Feed Forward Neural Network
2.   Convolutional Neural Network
3.   Resnet34 pretrained architecture
4.   Efficientnet-B4 pretrained architecture
5.   Resnext50_32x4d pretrained architecture


The project is inspired from the [Zero to GANs](https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans) course by the data science learning platform, [Jovian](https://www.jovian.ai).

## Data description
The dataset consist of 21,367 labeled images of cassava plant leaves collected during a regular survey in Uganda. Each image is an RGB image of size 600 x 800 pixels. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This is in a format that most realistically represents what farmers would need to diagnose in real life.

## Data exploration
Let us begin by downloading the dataset.
<pre> ```python
!pip install jovian opendatasets --upgrade --quiet
import opendatasets as od
#dataset_url='https://www.kaggle.com/aswiniabraham/cassava-leaf-disease-image-folders-600x800'

dataset_url='https://www.kaggle.com/c/cassava-leaf-disease-classification/data'
od.download(dataset_url) ``` </pre>
