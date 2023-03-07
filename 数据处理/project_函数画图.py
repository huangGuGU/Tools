import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 200)
y = 1 / (1 + np.exp(-100*(x - 0.404)))

plt.plot(x, y)
plt.show()
