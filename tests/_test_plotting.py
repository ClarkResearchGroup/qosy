from .context import qosy as qy
import numpy as np

def _test_plot_chain_lattice():
    N = 6
    orbital_names = ['A', 'B']
    chain_lattice = qy.lattice.chain(N, orbital_names, periodic=False)

    qy.plot(chain_lattice, with_labels=True)
    qy.show()

def _test_plot_kagome_lattice():
    N1 = 6
    N2 = 6
    kagome_lattice = qy.lattice.kagome(N1, N2, periodic_boundaries=(False, False))

    qy.plot(kagome_lattice, with_labels=True)
    qy.show()

_test_plot_chain_lattice()
_test_plot_kagome_lattice()
