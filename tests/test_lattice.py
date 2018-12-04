from .context import qosy as qy
import numpy as np

def test_chain_lattice():
    N = 6
    chain_lattice = qy.lattice.chain(N, ['A', 'B'])

    expected_orbital_names  = ['A', 'B']*N
    expected_atom_positions = [np.zeros(1), np.zeros(1)]*N

    expected_orbitals = zip(expected_atom_positions, expected_orbital_names)
    
    assert(len(chain_lattice._orbitals) == len(expected_orbitals))
    
def test_kagome_lattice():
    N1 = 1
    N2 = 2
    kagome_lattice  = qy.lattice.kagome(N1, N2)
    
    assert(len(kagome_lattice._orbitals) == N1*N2*3)
