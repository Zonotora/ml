import matplotlib.pyplot as plt
import numpy as np

n = 1000
n_train = 800

Xx = np.linspace(0, 10, n)
Y = Xx**2
X = np.zeros((n, 2))
X[:, 0] = Xx

np.random.seed(2)
indices = np.random.choice(n, n - n_train)
mask = np.ones(len(X), bool)
mask[indices] = 0

X_train = X[mask]
Y_train = Y[mask]
X_test = X[~mask]
Y_test = Y[~mask]

n_input = 2
n_hidden = 10
n_output = 1

w_i = np.random.randn(n_input, n_hidden) * 0.5
b_i = np.random.randn(n_hidden, 1) * 0.1
w_o = np.random.randn(n_hidden, 1) * 0.5
b_o = np.random.randn(n_output, 1) * 0.1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def forward(x):
    z_1 = w_i.T @ x + b_i
    A_1 = sigmoid(z_1)
    return w_o.T @ A_1 + b_o


for i in range(1000):
    for j in range(len(X_train)):
        y_true = Y_train[j] / 100

        # Forward pass
        x = X_train[j].reshape(2, 1)
        z_1 = w_i.T @ x + b_i
        A_1 = sigmoid(z_1)
        y = w_o.T @ A_1 + b_o

        # Backward pass
        # Loss fn = (y_hat - y)^2
        dL = -2 * (y_true - y)
        # Calculate in reverse
        # dL/db_o = dL/dy * dy/db_o
        db_o = dL * 1
        # dL/dw_o = dL/dy * dy/dw_o
        # either double transpose or element-wise
        dw_o = dL * A_1
        # dL/db_i = dL/dy * dy/dA_1 * dA_1/dz_1 * dz_1/db_i
        db_i = dL * w_o * d_sigmoid(z_1) * 1
        # dL/dw_i = dL/dy * dy/dA_1 * dA_1/dz_1 * dz_1/dw_i
        dw_i = (dL * w_o * d_sigmoid(z_1) @ x.T).T

        lr = 0.01

        w_i -= lr * dw_i
        b_i -= lr * db_i
        w_o -= lr * dw_o
        b_o -= lr * db_o

preds = []
for j in range(len(X_test)):
    y_pred = forward(X_test[j].reshape(2, 1))
    preds.append(y_pred[0][0] * 100)

plt.plot(Y_test)
plt.plot(preds)
plt.show()
