from .context import qosy as qy
import numpy as np

def test_kagome_orbital_relabeling():
    ### Without spin-1/2 orbital labels
    N1 = 2
    N2 = 2
    kagome_lattice = qy.lattice.kagome(N1, N2)

    string = ''
    for ind in range(3*N1*N2):
        string += ' '+str(ind)+'\n'

    new_string = qy.relabel_orbitals(string, kagome_lattice)

    expected_string ="""_{(0, 0); 0}
_{(0, 0); 1}
_{(0, 0); 2}
_{(0, 1); 0}
_{(0, 1); 1}
_{(0, 1); 2}
_{(1, 0); 0}
_{(1, 0); 1}
_{(1, 0); 2}
_{(1, 1); 0}
_{(1, 1); 1}
_{(1, 1); 2}
"""
    assert(new_string == expected_string)

    
    ### With spin-1/2 orbital labels
    kagome_lattice = qy.lattice.kagome(N1, N2, orbital_names=['Up', 'Dn'])

    string = ''
    for ind in range(3*2*N1*N2):
        string += ' '+str(ind)+'\n'

    new_string = qy.relabel_orbitals(string, kagome_lattice)

    expected_string ="""_{(0, 0); 0; Up}
_{(0, 0); 0; Dn}
_{(0, 0); 1; Up}
_{(0, 0); 1; Dn}
_{(0, 0); 2; Up}
_{(0, 0); 2; Dn}
_{(0, 1); 0; Up}
_{(0, 1); 0; Dn}
_{(0, 1); 1; Up}
_{(0, 1); 1; Dn}
_{(0, 1); 2; Up}
_{(0, 1); 2; Dn}
_{(1, 0); 0; Up}
_{(1, 0); 0; Dn}
_{(1, 0); 1; Up}
_{(1, 0); 1; Dn}
_{(1, 0); 2; Up}
_{(1, 0); 2; Dn}
_{(1, 1); 0; Up}
_{(1, 1); 0; Dn}
_{(1, 1); 1; Up}
_{(1, 1); 1; Dn}
_{(1, 1); 2; Up}
_{(1, 1); 2; Dn}
"""
    
    assert(new_string == expected_string)

    
