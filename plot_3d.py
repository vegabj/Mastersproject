from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_3d(Z, labelX, labelY, fig=None, cmap=cm.coolwarm):
    if not fig:
        fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(len(labelX))
    Y = np.arange(len(labelY))
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                           linewidth=0, antialiased=True, vmin = 0.0, vmax = 1.0)

    # Customize the z axis.
    ax.set_zlim(0.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Add labels for x and y axis
    ax.set_xlabel(labelX)
    ax.set_ylabel(labelY)

    plt.show()
