# automatic-image-annotation

##Installation
* [lda](https://pypi.python.org/pypi/lda)

## Data Preparation
* Get the IAPR-T12 dataset and the comments needs to be tokenized using vectorizeIART.py
* If using NUS dataset the image can be downloaded using datacollector.py which uses reservoir sampling to get random images as the dataset is too large further removes the corrupted images.

## Training and Testing

* First generate the SIFT features, image name, quantized SIFT features through vectorize.py for train and test datset.
* Use these features to get the annotations for all the image in an txt file named loss.txt.
