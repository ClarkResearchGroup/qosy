from .context import qosy as qy
import numpy as np

# TODO: finish, debug, test
def test_symmetrize_basis_D4():
    # D_4 symmetric square with for orbitals
    # 0,...,3 on the vertices of the square.

    pass
    """
    L = 4
    
    # Lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.0, 1.0])

    lattice_vectors = [a1, a2]

    positions = [-0.5*a1-0.5*a2, -0.5*a1+0.5*a2, 0.5*a1+0.5*a2, 0.5*a1-0.5*a2]
    
    unit_cell = qy.UnitCell(lattice_vectors)
    for pos in positions:
        unit_cell.add_atom(pos)
        
    lattice = qy.Lattice(unit_cell, (1,1), periodic_boundaries=(False,False))

    k = [1,2]
    basis = qy.cluster_basis(k, np.arange(L), 'Majorana')

    # Rotation
    t = 2.0*np.pi/L
    Rmat = np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t), np.cos(t)]])
    R = qy.space_group_symmetry(lattice, Rmat, np.zeros(2))
    
    # Reflection
    Smat = np.array([[1.0,  0.0],
                     [0.0, -1.0]])
    S = qy.space_group_symmetry(lattice, Smat, np.zeros(2))
    
    group_generators = [R, S]

    # The symmetrized basis.
    sym_basis = qy.symmetrize_basis(basis, group_generators)

    expected_operators = []
    # One-site operators
    orb_ops = ['A', 'B', 'D']
    for orb_op in orb_ops:
        coeffs     = np.ones(L)
        op_strings = [qy.opstring(orb_op+' {}'.format(site)) for site in range(L)]
        op = qy.Operator(coeffs, op_strings)
        op.normalize()

        expected_operators.append(op)

    
    # Two-site operators
    for bond_length in range(1,3):
        for ind_op1 in range(3):
            orb_op1 = orb_ops[ind_op1]
            for ind_op2 in range(ind_op1,3):
                orb_op2 = orb_ops[ind_op2]

                num_bonds = L
                # Hard-coded check for length 2 bonds.
                if bond_length == 2:
                    num_bonds = L//2
                
                if ind_op1 != ind_op2:
                    coeffs     = np.ones(2*num_bonds)
                    op_strings = [qy.opstring('{} {} {} {}'.format(orb_op1, np.minimum(site, (site+bond_length)%L), orb_op2, np.maximum(site, (site+bond_length)%L))) for site in range(num_bonds)] \
                                 + [qy.opstring('{} {} {} {}'.format(orb_op2, np.minimum(site, (site+bond_length)%L), orb_op1, np.maximum(site, (site+bond_length)%L))) for site in range(num_bonds)]
                else:
                    coeffs     = np.ones(num_bonds)
                    op_strings = [qy.opstring('{} {} {} {}'.format(orb_op1, np.minimum(site, (site+bond_length)%L), orb_op2, np.maximum(site, (site+bond_length)%L))) for site in range(num_bonds)]

                op = qy.Operator(coeffs, op_strings)
                op.normalize()
                
                expected_operators.append(op)

    print('sym_basis = ')
    for op in sym_basis:
        print(op)
    print('expected_operators = ')
    for op in expected_operators:
        print(op)
                
    same_operators = True
    for opA in sym_basis:
        operator_found = False
        for opB in expected_operators:
            if (opA-opB).norm() < 1e-12 or (opA+opB).norm() < 1e-12:
                operator_found = True
                break
        if not operator_found:
            same_operators = False
            print('Failed to find op: \n{}'.format(opA))
            break
        
    assert(same_operators)
    """

def test_symmetrize_basis_D6():
    # D_6 symmetric hexagon with six orbitals
    # 0,...,5 on the vertices of the hexagon.

    L = 6
    
    # Lattice spacing
    a  = 1.0
    # Lattice vectors
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([1.0/2.0, np.sqrt(3.0)/2.0])

    lattice_vectors = [a1, a2]
    
    # Positions of the three sites in a primitive unit cell.
    r1 = np.zeros(2)
    r2 = a1 / 2.0
    r3 = a2 / 2.0

    positions = [r3, r2, a1, r3+a1, r2+a2, a2]
    # Recenter so that center of hexagon is at origin.
    center_pos = -r2+a1+r3
    positions = [pos - center_pos for pos in positions]
    
    unit_cell = qy.UnitCell(lattice_vectors)
    for pos in positions:
        unit_cell.add_atom(pos)
        
    lattice = qy.Lattice(unit_cell, (1,1), periodic_boundaries=(False,False))
    
    basis = qy.cluster_basis([1,2], np.arange(L), 'Pauli')

    # Rotation
    t = 2.0*np.pi/L
    Rmat = np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t), np.cos(t)]])
    R = qy.space_group_symmetry(lattice, Rmat, np.zeros(2))
    
    # Reflection
    Smat = np.array([[-1, 0],
                     [0,  1]])
    S = qy.space_group_symmetry(lattice, Smat, np.zeros(2))
    
    group_generators = [R, S]

    # The symmetrized basis.
    sym_basis = qy.symmetrize_basis(basis, group_generators)

    expected_operators = []
    # One-site operators
    orb_ops = ['X', 'Y', 'Z']
    for orb_op in orb_ops:
        coeffs     = np.ones(L)
        op_strings = [qy.opstring(orb_op+' {}'.format(site)) for site in range(L)]
        op = qy.Operator(coeffs, op_strings)
        op.normalize()

        expected_operators.append(op)

    
    # Two-site operators
    for bond_length in range(1,4):
        for ind_op1 in range(3):
            orb_op1 = orb_ops[ind_op1]
            for ind_op2 in range(ind_op1,3):
                orb_op2 = orb_ops[ind_op2]

                num_bonds = L
                # Hard-coded check for length 3 bonds.
                if bond_length == 3:
                    num_bonds = L//2
                
                if ind_op1 != ind_op2:
                    coeffs     = np.ones(2*num_bonds)
                    op_strings = [qy.opstring('{} {} {} {}'.format(orb_op1, np.minimum(site, (site+bond_length)%L), orb_op2, np.maximum(site, (site+bond_length)%L))) for site in range(num_bonds)] \
                                 + [qy.opstring('{} {} {} {}'.format(orb_op2, np.minimum(site, (site+bond_length)%L), orb_op1, np.maximum(site, (site+bond_length)%L))) for site in range(num_bonds)]
                else:
                    coeffs     = np.ones(num_bonds)
                    op_strings = [qy.opstring('{} {} {} {}'.format(orb_op1, np.minimum(site, (site+bond_length)%L), orb_op2, np.maximum(site, (site+bond_length)%L))) for site in range(num_bonds)]

                op = qy.Operator(coeffs, op_strings)
                op.normalize()
                
                expected_operators.append(op)
            
    same_operators = True
    for opA in sym_basis:
        operator_found = False
        for opB in expected_operators:
            if (opA-opB).norm() < 1e-12 or (opA+opB).norm() < 1e-12:
                operator_found = True
                break
        if not operator_found:
            same_operators = False
            print('Failed to find op: \n{}'.format(opA))
            break
        
    assert(same_operators)
