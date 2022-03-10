import numpy as np
import matplotlib.pyplot as plt

# Generate data
# x: Study hours
# y: pass (1) or not (0)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.25, 4.50, 4.75, 5.00, 5.50, 5.75, 7.25, 8.25]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
             1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1])

# extended data
X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

n_sample = len(X)

print(n_sample)
# Initialize ththeta, alpha, eps
theta = 5
eps = 1e-4
d = X.shape[0]
w_init = np.random.randn(d, 1)


def sigmoid(z):
    return (1/(1+np.exp(-z)))


def logistic_sigmoid_regression(X, y, w_init, theta, eps=1e-4, max_iter=10000):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_iter:
        # mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + theta*(yi - zi)*xi
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < eps:
                    return w
            w.append(w_new)
    return w


w = logistic_sigmoid_regression(X, y, w_init, theta)
print(w[-1])

X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(X0, y0, 'ro')
plt.plot(X1, y1, 'bs')

xx = np.linspace(0, 10, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0/w1
yy = sigmoid(w0 + w1*xx)

plt.plot(xx, yy, 'g-', linewidth=2)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()
