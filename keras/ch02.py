from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("train_images shape >> ",train_images.shape)  # 60000 record with (28 × 28 pixels)
print("train_labels shape >> ",train_labels.shape)
print("test_images shape >> ",test_images.shape)
print("test_labels shape >> ",test_labels.shape)

from keras.models import Sequential
model = Sequential()

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Activation('softmax'))

network.add(layers.Dense(10, activation='softmax'))


