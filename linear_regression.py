import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class lr_model():

    def __init__(self):
        pass

    def initialize_model(self):
        pass

    def test(self):
        pass

    def train(self):
        pass

learning_rate = 0.001
n_epoch = 10000

n = 128
n_fit = 2
m, b = 2.0, 1.0 # slope, y-intercept
k_rand = 2.0
xr = [0,10]
x_train = np.random.rand(n,n_fit).astype(np.float32)*(xr[1]-xr[0])+xr[0]
y_train = m*x_train + (b+k_rand*(np.random.rand(n,n_fit)-0.5)).astype(np.float32)
file_model = os.path.join(os.path.dirname(__file__), "model", f"linear_regression_m{m:.4f}_b{b:.4f}.ckpt")
# initialize the model
model = nn.Linear(x_train.shape[1], x_train.shape[1])
if os.path.exists(file_model):
    # load existing model's dictionary
    print(f"loading existing model: {file_model}")
    model.load_state_dict(torch.load(file_model, weights_only=True))
# loss and optimizer functions
fn_loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
# train the model
x = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)
for epoch in range(n_epoch):
    # forward pass
    y_model = model(x)
    loss = fn_loss(y_model, y)
    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print (f"epoch [{epoch+1}/{n_epoch}], Loss: {loss.item():.4f}")

# test the model
x_test = np.random.rand(n,n_fit).astype(np.float32)*(xr[1]-xr[0])+xr[0]
y_pred = model(torch.from_numpy(x_test)).detach().numpy()
print(f"m={m:.4f} b={b:.4f}")
for i in range(n_fit):
    x = x_train[:,i]
    # training data
    i_sort = np.argsort(x)
    x = x[i_sort]
    y = (y_train[:,i])[i_sort]
    c = np.polyfit(x, y, 1)
    print(f" polyfit: m={c[0]:.4f} b={c[1]:.4f}")
    plt.plot(x, y, 'ro', label='Original data')
    # test data
    x = x_test[:,i]
    i_sort = np.argsort(x)
    x = x[i_sort]
    y = (y_pred[:,i])[i_sort]
    c = np.polyfit(x, y, 1)
    print(f" model: m={c[0]:.4f} b={c[1]:.4f}")
    plt.plot(x, y, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), file_model)