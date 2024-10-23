import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images to 1D vectors of size 28*28 (784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Add noise to the images for the purpose of enhancement
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip the noisy images to stay in the range [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Define the autoencoder model
input_img = layers.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

autoencoder = models.Model(input_img, decoded)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
history = autoencoder.fit(x_train_noisy, x_train,
                          epochs=20,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test_noisy, x_test))

# Evaluate the autoencoder
decoded_imgs = autoencoder.predict(x_test_noisy)

# Accuracy calculation
# Note: For an autoencoder, accuracy is not typically used, but you can calculate it based on reconstruction error
# For demonstration purposes, we'll use a simple metric based on the mean squared error between original and denoised images

from sklearn.metrics import mean_squared_error

# Flatten the images for mse calculation
x_test_flat = x_test.reshape(-1, 784)
decoded_imgs_flat = decoded_imgs.reshape(-1, 784)

mse = mean_squared_error(x_test_flat, decoded_imgs_flat)
print(f'Mean Squared Error: {mse}')

# Display original, noisy, and denoised images
def plot_images(original, noisy, denoised, num_images=10):
    plt.figure(figsize=(20, 6))
    for i in range(num_images):
        # Display original images
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')

        # Display noisy images
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy[i].reshape(28, 28), cmap='gray')
        ax.set_title("Noisy")
        ax.axis('off')

        # Display denoised images
        ax = plt.subplot(3, num_images, i + 1 + 2*num_images)
        plt.imshow(denoised[i].reshape(28, 28), cmap='gray')
        ax.set_title("Denoised")
        ax.axis('off')

    plt.show()

plot_images(x_test, x_test_noisy, decoded_imgs)
