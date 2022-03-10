import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.arange(-10, 10, 0.25)
n_sample = len(x)
# print(n_sample)
noise = np.random.normal(0, 1, n_sample)
y = -3*x + 5 + noise
plt.plot(x, y, 'ro')
# plt.show()

# Initialize theta, alpha, eps
theta_0 = np.random.randint(-10, 10)
theta_1 = np.random.randint(-5, 5)
alpha = 0.01
eps = 1e-4

# Update theta's values
# Check when to stop
iter = 1
while True:
    iter += 1
    theta_0 = theta_0 - alpha*np.mean(theta_1*x + theta_0 - y)
    theta_1 = theta_1 - alpha*np.mean((theta_1*x + theta_0 - y)*x)

    # Visualize updating process
    x_vis = np.array([-10.0, 10.0])
    y_vis = theta_1*x_vis + theta_0
    temp = plt.plot(x_vis, y_vis)
    plt.pause(0.0001)

    der_theta_0 = np.mean(theta_1*x + theta_0 - y)
    der_theta_1 = np.mean((theta_1*x + theta_0 - y)*x)

    if abs(der_theta_0) < eps and abs(der_theta_1) < eps:
        print('Finish update at iterate:', iter)
        break

print('Theta 0 value:', theta_0)
print('Theta 1 value:', theta_1)
plt.show()
