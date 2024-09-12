import tensorflow
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import numpy as np

# 1. Data Preparation

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Flatten the images to 1D vectors of size 784 (28*28)
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. Layerwise Pre-training

def build_autoencoder(input_dim, encoding_dim):
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder, encoder

# Layerwise training parameters
layer_dims = [512, 256, 128]
input_dim = x_train.shape[1]
autoencoders = []
encoders = []
encoded_input = x_train

# Train each layer's autoencoder
for encoding_dim in layer_dims:
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.fit(encoded_input, encoded_input,
                    epochs=10,
                    batch_size=256,
                    shuffle=True)
    encoded_input = encoder.predict(encoded_input)
    autoencoders.append(autoencoder)
    encoders.append(encoder)
    input_dim = encoding_dim

# 3. Model Definition and Initialization

# Define the DNN with pre-trained weights
model = Sequential()

# Add pre-trained layers
input_dim = x_train.shape[1]
for encoder in encoders:
    dense_layer = Dense(encoder.layers[1].units, activation='relu', input_shape=(input_dim,))
    model.add(dense_layer)
    dense_layer.set_weights(encoder.layers[1].get_weights())
    input_dim = encoder.layers[1].units

# Add the output layer
model.add(Dense(10, activation='softmax'))

# 4. Fine-tuning

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}\n')

# Predict on the test set
predictions = model.predict(x_test)
print(f'Predicted label: {np.argmax(predictions[0])}, True label: {np.argmax(y_test[0])}')