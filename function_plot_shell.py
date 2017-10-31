
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def fsa_acceptance_probability(t, d):

    return(np.exp(-d / t))


fig = plt.figure()
ax = fig.gca()

t_max = 1
d_max = 1
grid = 0.0001

# Make data.
t = np.arange(0, t_max, grid)
d = np.arange(0, d_max, grid)


t, d = np.meshgrid(t, d)
Z = fsa_acceptance_probability(t, d)

im = plt.imshow(Z, origin='lower', extent=[0, t_max, 0, d_max])

ax.set_xlabel('t')

ax.set_ylabel('d')


plt.show()
