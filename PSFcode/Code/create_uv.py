import numpy as np

def pairwise_baselines(ants):
    """Return all plus/minus baseline vectors between antenna pairs."""
    baselines = []
    # loop over each antenna i: [x,y]
    for i in range(len(ants)):
        # loop over each other antenna j = i+1 such that we don't count any pair twice
        for j in range(i+1, len(ants)):
            # vector v = [delta_x, delta_y]
            v = ants[j] - ants[i]
            baselines.append(v)
            # baselines.append(-v)
    return np.array(baselines)

array = np.array(
[[-0.38707748, -0.22950169],
[0.25416738, -0.20148773],
[0.06722677, -0.44495007],
[-0.23054176, -0.38645892],
[-0.34811393,  0.28516081],
[0.38254836,  0.23697415],
[-0.07422633,  0.44383606],
[0.43415994, -0.11834334],
[-0.44789622, -0.04346234]])

from matplotlib import pyplot as plt
for i, ant in enumerate(array):
    plt.scatter(*ant, label = f'ant{i}')
plt.axis('equal')
plt.legend()
plt.show()

uv = pairwise_baselines(array)
baselines = np.linalg.norm(uv, axis = 1)
print(baselines)
np.savetxt('baselines.txt', baselines, fmt = '%.2e')