import numpy as np
from tensorflow import keras
from keras.datasets import fashion_mnist

#loading the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#normalizing the data
X_train_norm = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
X_test_norm = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

print(X_train_norm.shape)
