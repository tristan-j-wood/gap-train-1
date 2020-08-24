import matplotlib.pyplot as plt
from gaptrain.exceptions import PlottingFailed
from matplotlib.colors import LogNorm
import matplotlib as mpl
import numpy as np

mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['axes.linewidth'] = 1


def histogram(energies=None, forces=None, name=None, relative_energies=True):
    """
    Plot a histogram of energies, forces or both

    :param energies: (list(float))
    :param forces: (list(float))
    :param name: (str) or None
    ;:param relative_energies: (bool)
    """
    assert energies is not None or forces is not None
    fig, ax = fig_ax(energies, forces)

    if energies is not None:

        if len(energies) == 0:
            raise PlottingFailed('No energies')

        ax_e = ax if forces is None else ax[0]

        if relative_energies:
            energies = np.array(energies) - min(energies)

        ax_e.hist(energies,
                  bins=np.linspace(min(energies), max(energies), 30),
                  alpha=0.5,
                  edgecolor='darkblue',
                  linewidth=0.2
                  )

        # Energy histogram formatting
        ax_e.set_xlabel('Energy / eV')
        ax_e.set_ylabel('Frequency')

    if forces is not None:
        if len(forces) == 0:
            raise PlottingFailed('No energies')

        ax_f = ax if energies is None else ax[1]

        ax_f.hist(forces,
                  bins=np.linspace(min(forces), max(forces), 50),
                  color='orange',
                  alpha=0.5,
                  edgecolor='darkorange',
                  linewidth=0.2)

        # Force histogram formatting
        ax_f.set_xlabel('|$F$| / ev Å$^{-1}$')
        ax_f.set_ylabel('Frequency')

    return show_or_save(name)


def correlation(true_energies=None,
                predicted_energies=None,
                true_forces=None,
                predicted_forces=None,
                name=None):
    """
    Plot a correlation plot between predicted energies and/or forces and their
    ground truth values

    :param true_energies: (list(float))
    :param predicted_energies: (list(float))
    :param true_forces: (list(float))
    :param predicted_forces: (list(float))
    :param name: (str) or None
    """
    if true_energies is not None:
        assert predicted_energies is not None

    if true_forces is not None:
        assert predicted_forces is not None

    fig, ax = fig_ax(true_energies, true_forces)

    if true_energies is not None:
        ax_e = ax if true_forces is None else ax[0]
        # Scatter the true and predicted data
        ax_e.scatter(true_energies, predicted_energies)

        # Plot a y = x line
        all_energies = [e for energies in (true_energies, predicted_energies)
                        for e in energies]

        min_e, max_e = min(all_energies), max(all_energies)
        delta = np.abs(max_e - min_e)/50
        pair = [min_e - delta, max_e + delta]

        ax_e.plot(pair, pair, lw=0.5, c='k', zorder=0)
        ax_e.set_xlim(*pair)
        ax_e.set_ylim(*pair)

        # Energy plot formatting
        ax_e.set_xlabel('True Energy / eV', size=12)
        ax_e.set_ylabel('Predicted Energy / eV', size=12)

    if true_forces is not None:
        ax_f = ax if true_energies is None else ax[1]

        all_forces = [f for forces in (true_forces, predicted_forces)
                      for f in forces]

        min_f, max_f = min(all_forces), max(all_forces)

        # Histogram the forces in 2D
        hist = ax_f.hist2d(true_forces, predicted_forces,
                           density=True,
                           bins=[np.linspace(min_f, max_f, 200),
                                 np.linspace(min_f, max_f, 200)],
                           norm=LogNorm())

        delta = np.abs(max_f - min_f)/50
        pair = [min_f - delta, max_f + delta]

        # y = x and extended limits
        ax_f.plot(pair, pair, lw=0.5, c='k')
        ax_f.set_xlim(*pair)
        ax_f.set_ylim(*pair)

        fig.colorbar(hist[3], ax=ax_f)

        # Energy plot formatting
        ax_f.set_xlabel('True Force / eV Å$^{-1}$')
        ax_f.set_ylabel('Predicted Force / eV Å$^{-1}$')

    return show_or_save(name)


def show_or_save(name):
    """If name is None then show the plot otherwise save it as a .png"""
    plt.tight_layout()

    if name is None:
        plt.show()

    else:
        plt.savefig(f'{name}.png', dpi=300)

    plt.close()
    return None


def fig_ax(energies, forces):
    """Get the appropriate axes for a set of energies and forces"""

    if energies is not None and forces is not None:
        size = (10, 4.5)
        cols = 2

    else:
        size = (4.5, 4.5)
        cols = 1

    return plt.subplots(nrows=1, ncols=cols, figsize=size)
