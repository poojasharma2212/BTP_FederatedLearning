import numpy as np
import matplotlib.pyplot as plt
 
# data to be plotted
x = np.arrange(1,25)
 y =x
# plotting
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color ="red")
plt.show()