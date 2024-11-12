
Back Propogation Algorithm
==

## Introduction

This code implements a simple neural network with one hidden layer. The network is trained using the mean squared error (MSE) loss function and the sigmoid activation function.

---
## Loading Data

```python
data = pd.read_csv('data.csv')
X = data[['input1', 'input2']].values
y = data['output'].values.reshape(-1, 1)
```

The code loads a CSV file containing the input data and the corresponding output values. The input data is stored in the `X` variable, and the output values are stored in the `y` variable.

---
## Neural Network Parameters

```python
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
```

The code defines the neural network parameters:

*   `input_size`: The number of input features (2 in this case).
*   `hidden_size`: The number of neurons in the hidden layer (2 in this case).
*   `output_size`: The number of output features (1 in this case).
*   `learning_rate`: The learning rate for the gradient descent algorithm (0.1 in this case).

---
## Weights and Biases

```python
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))
```

The code initializes the weights and biases for the neural network:

*   `W1`: The weights for the connections between the input layer and the hidden layer.
*   `b1`: The biases for the hidden layer.
*   `W2`: The weights for the connections between the hidden layer and the output layer.
*   `b2`: The biases for the output layer.

---
## Activation Function

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
```

The code defines the sigmoid activation function and its derivative:

*   `sigmoid(x)`: The sigmoid function, which maps the input `x` to a value between 0 and 1.
*   `sigmoid_derivative(x)`: The derivative of the sigmoid function, which is used in the backpropagation algorithm.

---
## Forward Pass

```python
def forward_pass(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2
```

The code defines the forward pass through the neural network:

*   `z1 = np.dot(X, W1) + b1`: The weighted sum of the input values and the weights for the connections between the input layer and the hidden layer, plus the biases for the hidden layer.
*   `a1 = sigmoid(z1)`: The output of the hidden layer, which is the sigmoid of the weighted sum.
*   `z2 = np.dot(a1, W2) + b2`: The weighted sum of the output values of the hidden layer and the weights for the connections between the hidden layer and the output layer, plus the biases for the output layer.
*   `a2 = sigmoid(z2)`: The output of the output layer, which is the sigmoid of the weighted sum.

---
## Backward Pass

```python
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
```

The code defines the backward pass through the neural network:

*   `dz2 = a2 - y`: The error between the predicted output and the actual output.
*   `dW2 = np.dot(a1.T, dz2) / m`: The gradient of the loss function with respect to the weights for the connections between the hidden layer and the output layer.
*   `db2 = np.sum(dz2, axis=0, keepdims=True) / m`: The gradient of the loss function with respect to the biases for the output layer.
*   `dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)`: The error between the predicted output of the hidden layer and the actual output of the hidden layer.
*   `dW1 = np.dot(X.T, dz1) / m`: The gradient of the loss function with respect to the weights for the connections between the input layer and the hidden layer.
*   `db1 = np.sum(dz1, axis=0, keepdims=True) / m`: The gradient of the loss function with respect to the biases for the hidden layer.

---
## Training the Neural Network

```python
epochs = 10000
for epoch in range(epochs):
    z1, a1, z2, a2 = forward_pass(X)
    backward_pass(X, y, z1, a1, z2, a2)
    if epoch % 1000 == 0:
        loss = np.mean((a2 - y) ** 2)
        print(f'Epoch {epoch}, Loss: {loss}')
```

The code trains the neural network using the mean squared error (MSE) loss function and the gradient descent algorithm:

*   `epochs = 10000`: The number of epochs to train the neural network.
*   `for epoch in range(epochs)`: The loop that trains the neural network for the specified number of epochs.
*   `z1, a1, z2, a2 = forward_pass(X)`: The forward pass through the neural network.
*   `backward_pass(X, y, z1, a1, z2, a2)`: The backward pass through the neural network.
*   `if epoch % 1000 == 0`: The condition that checks if the current epoch is a multiple of 1000.
*   `loss = np.mean((a2 - y) ** 2)`: The calculation of the MSE loss function.
*   `print(f'Epoch {epoch}, Loss: {loss}')`: The print statement that displays the current epoch and the MSE loss function.

---
## Testing the Neural Network

```python
def predict(X):
    _, _, _, a2 = forward_pass(X)
    return a2

print("Predictions:")
print(predict(X))
```

The code defines a function that predicts the output of the neural network for a given input:

*   `def predict(X)`: The function that predicts the output of the neural network.
*   `_, _, _, a2 = forward_pass(X)`: The forward pass through the neural network.
*   `return a2`: The return statement that returns the predicted output.
*   `print("Predictions:")`: The print statement that displays the predicted output.
*   `print(predict(X))`: The print statement that displays the predicted output for the given input.
*   `_, _, _, a2 = forward_pass(X)`: The forward pass through the neural network.
*   `return a2`: The return statement that returns the predicted output.
*   `print("Predictions:")`: The print statement that displays the predicted output.
*   `print(predict(X))`: The print statement that displays the predicted output for the given input.
  
Complete Code
--
```python
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
```