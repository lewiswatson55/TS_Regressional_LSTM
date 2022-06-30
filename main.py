# Time Series Regressional Prediction Model using LSTM Neural Network PyTorch

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Create random dataset from random numbers
X_train = np.random.rand(100, 1)
y_train = 2 * X_train + 1
X_test = np.random.rand(50, 1)
y_test = 2 * X_test + 1

# Create the class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.linear(output[-1])
        return output, hidden
    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_size)),
                Variable(torch.zeros(1, 1, self.hidden_size)))

# Create the 2d model
model = LSTM(1, 10, 1)

# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create the loss function
loss_func = nn.MSELoss()

# Create the hidden state
hidden = model.init_hidden()

# Create the input
input = Variable(torch.from_numpy(X_train))

# print input shape
print(input.shape)

# Create the output
output = Variable(torch.from_numpy(y_train))

# Train the model
for epoch in range(1000):
    # Forward pass
    output, hidden = model(input, hidden)
    loss = loss_func(output, output)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0: # this is to print the loss every 100 epochs
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 1000, loss.data[0]))

#   Test the model
# input = Variable(torch.from_numpy(X_test))
# output = model(input, hidden)
# pred = output.data.numpy()

# Plot the results





