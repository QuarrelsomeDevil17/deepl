import tensorflow as tf
from tensorflow.keras.models import Sequential #type: ignore
from keras.preprocessing.sequence import pad_sequences #type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Activation #type: ignore
from tensorflow.keras.datasets import imdb #type: ignore
from tensorflow.keras.preprocessing import sequence #type: ignore
import matplotlib.pyplot as plt

# Step 2: Load and Preprocess the IMDB Dataset
max_features = 15000  # Only consider the top 10,000 words in the dataset
max_len = 100  # Cut reviews after 500 words
batch_size = 128

# Load IMDB dataset and only keep max_features most frequent words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure equal input length
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Step 3: Build the RNN Model with Multiple Layers
model = Sequential([
    Embedding(max_features, 32, input_length=max_len),  # Embedding layer
    SimpleRNN(16, return_sequences=True),  # First RNN layer
    SimpleRNN(8),  # Second RNN layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Step 4: Compile the Model with Accuracy Metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(x_train, y_train, epochs=8, batch_size=batch_size, validation_split=0.2)

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test Accuracy: {test_acc:.3f}')