import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical

# Step 2: Importing libraries and loading dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Print the shape of the dataset
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Step 3: Preprocessing the dataset
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 4: Model definition and training
cnn = tf.keras.models.Sequential()

# Initialize convolutional Network
cnn.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())

# Add Artificial Neural Network
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='relu'))

# Compiling CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training CNN on the training set and evaluation on the testing dataset
cnn.fit(x=x_train, validation_data=x_test, epochs=5)

# Evaluating Results