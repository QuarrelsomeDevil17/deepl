import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# Step 1: Dataset Preparation
# Assuming you have a dataset of images (e.g., MNIST), load and preprocess them
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

def build_generator_cnn():
    model = models.Sequential([
        # Start with a fully connected layer to interpret the seed
        layers.Dense(7*7*128, input_dim=100, activation='relu'),
        layers.Reshape((7, 7, 128)),  # Reshape into an image format

        # Upsample to 14x14
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),

        # Upsample to 28x28
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),

        # Output layer with the shape of the target image, 1 channel for grayscale
        layers.Conv2D(1, kernel_size=7, activation='sigmoid', padding='same')
    ])
    return model

def build_discriminator_cnn():
    model = models.Sequential([
        # Input layer with the shape of the target image
        layers.Conv2D(64, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding='same', activation='relu'),

        # Downsample to 14x14
        layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),

        # Further downsampling and flattening to feed into a dense output layer
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Instantiate the CNN-based Generator and Discriminator
generator_cnn = build_generator_cnn()
discriminator_cnn = build_discriminator_cnn()

# Compile the Discriminator
discriminator_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])

# Set the Discriminator's weights to non-trainable (important when we train the combined GAN model)
discriminator_cnn.trainable = False

# Combined GAN model with CNN
gan_input = layers.Input(shape=(100,))
gan_output = discriminator_cnn(generator_cnn(gan_input))
gan_cnn = models.Model(gan_input, gan_output)
gan_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

epochs = 10000
batch_size = 64

for epoch in range(epochs):
    ############################
    # 1. Train the Discriminator
    ############################

    # Generate batch of noise
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_images = generator_cnn.predict(noise)

    # Get a random batch of real images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]

    # Labels for generated and real data
    fake_labels = np.zeros((batch_size, 1))
    real_labels = np.ones((batch_size, 1))

    # Train the Discriminator (real classified as ones and generated as zeros)
    d_loss_real = discriminator_cnn.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator_cnn.train_on_batch(generated_images, fake_labels)

    #################################
    # 2. Train the Generator (via GAN)
    #################################

    # Train the generator (note that we want the Discriminator to mistake images as real)
    noise = np.random.normal(0, 1, (batch_size, 100))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan_cnn.train_on_batch(noise, valid_labels)

    # Plot the progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}")

    # Optionally, save generated images and display
    if epoch % 1000 == 0:
        generated_image = generator_cnn.predict(np.random.normal(0, 1, (1, 100)))
        plt.imshow(generated_image[0, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.show()
        plt.close()
