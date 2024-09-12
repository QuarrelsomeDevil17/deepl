import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

class AdaGradOptimizer:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.grad_squares = {}

    def update(self, param, grad):
        if param not in self.grad_squares:
            self.grad_squares[param] = np.zeros_like(grad)
        self.grad_squares[param] += np.square(grad)
        adjusted_grad = grad / (np.sqrt(self.grad_squares[param]) + self.epsilon)
        return self.learning_rate * adjusted_grad

class DNN:
    def __init__(self, layer_sizes):
        self.params = {}
        self.layer_sizes = layer_sizes
        self.initialize_params()

    def initialize_params(self):
        for i in range(1, len(self.layer_sizes)):
            self.params[f'W{i}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(2.0 / self.layer_sizes[i-1])
            self.params[f'b{i}'] = np.zeros((self.layer_sizes[i], 1))

    def forward(self, X):
        cache = {'A0': X}
        A = X
        for i in range(1, len(self.layer_sizes)):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            Z = np.dot(W, A) + b
            A = np.maximum(0, Z)  # ReLU activation
            cache[f'Z{i}'] = Z
            cache[f'A{i}'] = A
        return A, cache

    def compute_loss(self, A, Y):
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(A + 1e-8)) / m
        return loss

    def backward(self, cache, Y):
        grads = {}
        m = Y.shape[1]
        A_final = cache[f'A{len(self.layer_sizes)-1}']
        dA = - (np.divide(Y, A_final + 1e-8) - np.divide(1 - Y, 1 - A_final + 1e-8))
        for i in reversed(range(1, len(self.layer_sizes))):
            dZ = dA * (cache[f'Z{i}'] > 0)
            dW = np.dot(dZ, cache[f'A{i-1}'].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.dot(self.params[f'W{i}'].T, dZ)
            grads[f'dW{i}'] = dW
            grads[f'db{i}'] = db
        return grads

    def update_params(self, grads, optimizer):
        for i in range(1, len(self.layer_sizes)):
            self.params[f'W{i}'] -= optimizer.update(f'W{i}', grads[f'dW{i}'])
            self.params[f'b{i}'] -= optimizer.update(f'b{i}', grads[f'db{i}'])

    def train(self, X, Y, epochs=1000, learning_rate=0.01, batch_size=64):
        optimizer = AdaGradOptimizer(learning_rate=learning_rate)
        m = X.shape[1]
        loss_history = []

        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i + batch_size]
                Y_batch = Y_shuffled[:, i:i + batch_size]

                A, cache = self.forward(X_batch)
                loss = self.compute_loss(A, Y_batch)
                grads = self.backward(cache, Y_batch)
                self.update_params(grads, optimizer)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
            loss_history.append(loss)

        # Plot loss history
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

# Load and preprocess MNIST data
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
train_X = train_X.reshape(train_X.shape[0], -1).T / 255.0  # Flatten and normalize
test_X = test_X.reshape(test_X.shape[0], -1).T / 255.0  # Flatten and normalize
train_Y = np.eye(10)[train_Y].T  # One-hot encode
test_Y = np.eye(10)[test_Y].T  # One-hot encode

# Initialize and train the model
dnn = DNN([784, 128, 64, 10])
dnn.train(train_X, train_Y, epochs=1000, learning_rate=0.1, batch_size=64)
