import tensorflow as tf
from tensorflow.keras.models import Sequential #type: ignore
from keras.preprocessing.sequence import pad_sequences #type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Activation #type: ignore
from tensorflow.keras.datasets import imdb #type: ignore
from tensorflow.keras.preprocessing import sequence #type: ignore
import matplotlib.pyplot as plt

num_words = 15000
(X_train,Y_train),(X_test,Y_test) = imdb.load_data(num_words=num_words)

maxlen=130
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

rnn = Sequential()
rnn.add(Embedding(num_words,32,input_length =len(X_train[0]))) # num_words=15000
rnn.add(SimpleRNN(16,input_shape = (num_words,maxlen), return_sequences=False,activation="relu"))
rnn.add(Dense(1)) #flatten
rnn.add(Activation("sigmoid")) #using sigmoid for binary classification

#print(rnn.summary())
rnn.compile(loss="binary_crossentropy",optimizer="rms",metrics=["accuracy"])
history = rnn.fit(X_train,Y_train,validation_data = (X_test,Y_test),epochs = 5,batch_size=128,verbose = 1)
score = rnn.evaluate(X_test,Y_test)
print("accuracy:", score[1]*100)

plt.figure()
plt.plot(history.history["accuracy"],label="Train");
plt.plot(history.history["val_accuracy"],label="Test");
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show();

plt.figure()
plt.plot(history.history["loss"],label="Train");
plt.plot(history.history["val_loss"],label="Test");
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show();