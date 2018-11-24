from .context import qosy as qy
import numpy as np

def test_chain_lattice():
    N = 6
    chain_lattice = qy.lattice.chain(N)

    expected_labels    = [(1,0), (1,1), (1,2), (1,3), (1,4), (1,5)]
    expected_positions = np.zeros((1,N))
    expected_positions[0,:] = np.arange(N)
    
    assert(chain_lattice.labels == expected_labels)
    assert(np.allclose(chain_lattice.positions, expected_positions))
    
def test_kagome_lattice():
    N1 = 1
    N2 = 2
    boundary = ('Periodic','Open')
    kagome_lattice = qy.lattice.kagome(N1, N2, boundary=boundary)

    expected_labels    = [(1,0,0), (1,0,1), (2,0,0), (2,0,1), (3,0,0), (3,0,1)]    

    assert(len(kagome_lattice.labels) == N1*N2*3)
    assert(kagome_lattice.labels == expected_labels)

def test_decorated_lattice():
    N = 3
    chain_lattice = qy.lattice.chain(N)

    new_labels = ['A', 'B']
    decorated_chain_lattice = qy.lattice.decorate(chain_lattice, new_labels)

    expected_labels = [(1,0,'A'), (1,1,'A'), (1,2,'A'), (1,0,'B'), (1,1,'B'), (1,2,'B')]
    
    assert(set(decorated_chain_lattice.labels) == set(expected_labels))
    assert(int(decorated_chain_lattice.positions.shape[1]) == len(expected_labels)) 
