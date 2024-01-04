'''
https://medium.com/artificialis/get-started-with-computer-vision-by-building-a-digit-recognition-model-with-tensorflow-b2216823b90a
'''

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras import layers


# import dataset of 60,000 28x28 images of handrawn digits
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Optional code to display an example image from the dataset
'''
random_image = random.randint(0,  len(X_train))
plt.figure(figsize=(3, 3))
plt.imshow(X_train[random_image], cmap="gray")
plt.title(y_train[random_image])
plt.axis(False);
'''

#This model required input to be of the form [height, width, colorchannels]
#Since we are providing grayscale input, we set colorchannels to 1
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1, ))
X_train.shape # (60000, 28, 28, 1)

# Normalize data so that input is in range [0,1]
# Convert to float32 type
X_train = X_train / 255.
X_test = X_test / 255.
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# We follow TinyVGG architecture
model = tf.keras.Sequential([layers.Conv2D(filters=10, kernel_size=3,activation="relu", input_shape=(28,  28,  1)), layers.Conv2D(10,  3, activation="relu"),layers.MaxPool2D(),layers.Conv2D(10,  3, activation="relu"),layers.Conv2D(10,  3, activation="relu"),layers.MaxPool2D(),layers.Flatten(),layers.Dense(10, activation="softmax")])

# Print summary, compile and evauluate model
model.summary()
model.compile(loss="sparse_categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test, y_test)

# Save model
model.save("digit-recognizer.h5")
