import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.arange(-10, 10, 0.25)
n_sample = len(x)
# print(n_sample)
noise = np.random.normal(0, 1, n_sample)
Y = -3*x + 5 + noise
plt.plot(x, Y, 'ro')

_one = np.ones((1, n_sample))
X = np.concatenate((_one, [x]))
# print(X)

# Initialize theta, alpha, eps
theta = np.array([[-10], [-6]])
alpha = 0.01
eps = 1e-4

# Update theta's values
# Check when to stop
_iter = 1
while True:
    _iter += 1
    nabla = (1.0/n_sample) * np.dot(X, (np.dot(theta.T, X) - Y).T)
    theta = theta - alpha * nabla

    # Visualize updating process
    x_vis = np.array([-10.0, 10.0])
    y_vis = theta[1][0]*x_vis + theta[0][0]
    temp = plt.plot(x_vis, y_vis)
    plt.pause(0.0001)

    nabla = (1.0/n_sample) * np.dot(X, (np.dot(theta.T, X) - Y).T)
    if abs(nabla[0][0]) < eps and abs(nabla[1][0]) < eps:
        print('Finished update at iterate:', _iter)
        break

print('Theta value:', theta)

plt.show()
