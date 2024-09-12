import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images to 1D vectors of size 28*28 (784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert labels to categorical one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the model using Dense layers
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model with sgd optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=[
                  tf.keras.metrics.Precision(),
                  tf.keras.metrics.Recall()
                  ])

 # Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
results = model.evaluate(x_test, y_test)
test_loss = results[0]
test_precision = results[1]
test_recall = results[2]
print(f'Test loss: {test_loss}')
print(f'Test precision: {test_precision}')
print(f'Test recall: {test_recall}')