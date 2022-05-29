# MLP-Mixer-Style-Model-Fashion-MNIST
A unique MLP architecture (similar to that of MLP-Mixer) for classifying the fashion MNIST dataset, constructed as part of the neural networks and deep learning module. Built using the PyTorch machine learning framework.

Dataset used - Fashion MNIST (https://www.kaggle.com/datasets/zalando-research/fashionmnist).

## Architecture
* Involves three key componenets - the stem, backbone and classifier
* Stem - splits 28x28 image into 16 non-overlapping patches, each of dimensions 7x7 which are then vectorised (linear embedding)
* Backbone - three identical blocks, each comprising two MLPs. Using transpositions, the first MLP acts across patches, and the second acts on features within patches
* Classifier - computes a mean feature (across all spatial locations of patches) then maps to 10 outputs using Softmax regression
* Notable features - data augmentation (random horizontal flip, rotation and normalisation), layer normalisation, GELU activation, skip connections (between blocks)

## Final Model Accuracy
* Obtained 90%+ after 40 epochs on the validation set
