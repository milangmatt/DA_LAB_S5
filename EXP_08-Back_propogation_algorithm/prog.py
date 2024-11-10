import numpy as np
import pandas as pd

# Load CSV data
data = pd.read_csv('data.csv')
X = data[['input1', 'input2']].values
y = data['output'].values.reshape(-1, 1)

# Initialize neural network parameters
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1

# Weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Forward pass
def forward_pass(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backward pass
def backward_pass(X, y, z1, a1, z2, a2):
    global W1, b1, W2, b2
    m = X.shape[0]
    
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    
    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Training the neural network
epochs = 10000
for epoch in range(epochs):
    z1, a1, z2, a2 = forward_pass(X)
    backward_pass(X, y, z1, a1, z2, a2)
    if epoch % 1000 == 0:
        loss = np.mean((a2 - y) ** 2)
        print(f'Epoch {epoch}, Loss: {loss}')

# Testing the neural network
def predict(X):
    _, _, _, a2 = forward_pass(X)
    return a2

# Test the network
print("Predictions:")
print(predict(X))