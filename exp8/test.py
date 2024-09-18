import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


# 1. Load and Preprocess the IMDB Dataset
vocab_size = 10000  # Only consider the top 10,000 words in the dataset
max_length = 100    # Max length of each review
trunc_type = 'post'
padding_type = 'post'
# Load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform input length
x_train = pad_sequences(x_train, maxlen=max_length, padding=padding_type, truncating=trunc_type)
x_test = pad_sequences(x_test, maxlen=max_length, padding=padding_type, truncating=trunc_type)



# Explicitly build the model by calling build() with input shape
model.build(input_shape=(None, max_length))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Show the model summary
model.summary()

# 3. Train the Model
num_epochs = 10  # Number of epochs
history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test), verbose=2)

# 4. Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")