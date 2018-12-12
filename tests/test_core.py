#!/usr/bin/env python
from .context import qosy as qy
import numpy as np
import numpy.linalg as nla

def test_disordered_heisenberg():
    # Check that you always find the generators of the
    # global SU(2) symmetry, namely O_a = \sum_i \sigma^a_i
    # for a=x,y,z, for disordered SU(2) invariant
    # Heisenberg models.

    N = 6
    # Use a basis of one-site operators.
    basis = qy.cluster_basis(1, np.arange(N), 'Pauli')

    # Total X, total Y, and total Z expressed as vectors
    # in this basis.
    totalX = np.array([[op_string.orbital_operators == ['X']] for op_string in basis], dtype=complex)
    totalY = np.array([[op_string.orbital_operators == ['Y']] for op_string in basis], dtype=complex)
    totalZ = np.array([[op_string.orbital_operators == ['Z']] for op_string in basis], dtype=complex)
    
    generators = np.hstack((totalX, totalY, totalZ))
    
    np.random.seed(42)
    num_trials = 10
    for ind_trial in range(num_trials):
        heisenberg_model = qy.Operator(op_type='Pauli')
        for site1 in range(N):
            for site2 in range(site1+1,N):
                bond_coeffs = (2.0*np.random.rand() - 1.0) * np.ones(3)
                bond_ops    = [qy.opstring('X {} X {}'.format(site1,site2)), \
                               qy.opstring('Y {} Y {}'.format(site1,site2)), \
                               qy.opstring('Z {} Z {}'.format(site1,site2))]
                heisenberg_model += qy.Operator(bond_coeffs, bond_ops)

        commuting_ops = qy.commuting_operators(basis, heisenberg_model)
        
        assert(int(commuting_ops.shape[1]) == 3)
        
        overlaps = np.dot(np.conj(generators.T), commuting_ops)
        
        (eigvals, eigvecs) = nla.eigh(overlaps)
        inds_null_space    = np.where(np.abs(eigvals) < 1e-12)[0]

        assert(len(inds_null_space) == 0)

def test_disordered_heisenberg_operators():
    # Same as test_disordered_heisenberg, but using a list of Operators
    # as a "basis" instead of a Basis of OperatorStrings.

    N = 6
    # Use a basis of one-site operators.
    basis = qy.cluster_basis(1, np.arange(N), 'Pauli')

    # These are my operators O_1, O_2, O_3, O_4,
    # which form my list of Operators that I will use as a basis.
    totalX = qy.Operator(np.ones(N), [qy.opstring('X {}'.format(site)) for site in range(N)])
    totalY = qy.Operator(np.ones(N), [qy.opstring('Y {}'.format(site)) for site in range(N)])
    totalZ = qy.Operator(np.ones(N), [qy.opstring('Z {}'.format(site)) for site in range(N)])
    random_op = qy.Operator([1.0], [qy.opstring('X 0 Z {}'.format(N-1))]) # An extra unnecessary operator

    operators = [totalX, totalY, totalZ, random_op]

    # In the list of operators "basis", the expected answers
    # are the unit vectors [1,0,0], [0,1,0], and [0,0,1].
    expected_commuting_ops = np.zeros((len(operators),3), dtype=complex)
    expected_commuting_ops[0:3,0:3] = np.eye(3)
    
    np.random.seed(42)
    num_trials = 10
    for ind_trial in range(num_trials):
        heisenberg_model = qy.Operator(op_type='Pauli')
        for site1 in range(N):
            for site2 in range(site1+1,N):
                bond_coeffs = (2.0*np.random.rand() - 1.0) * np.ones(3)
                bond_ops    = [qy.opstring('X {} X {}'.format(site1,site2)), \
                               qy.opstring('Y {} Y {}'.format(site1,site2)), \
                               qy.opstring('Z {} Z {}'.format(site1,site2))]
                heisenberg_model += qy.Operator(bond_coeffs, bond_ops)

        commuting_ops = qy.commuting_operators(operators, heisenberg_model)
        
        assert(int(commuting_ops.shape[1]) == 3)
        
        overlaps = np.dot(np.conj(expected_commuting_ops.T), commuting_ops)
        
        (eigvals, eigvecs) = nla.eigh(overlaps)
        inds_null_space    = np.where(np.abs(eigvals) < 1e-12)[0]

        assert(len(inds_null_space) == 0)

def test_translation_invariance_chain():
    # Check that you find translationally invariant operators
    # when using the translation operator with invariant_operators()

    N = 6
    # Use a basis of one-site operators.
    basis = qy.cluster_basis(1, np.arange(N), 'Pauli')

    # Shift all labels to the right by one.
    permutation = np.concatenate((np.array([N-1]), np.arange(N-1)))

    # Translation operator
    P = qy.label_permutation(permutation)

    invariant_ops = qy.invariant_operators(basis, P)

    # Total X, total Y, and total Z expressed as vectors
    # in this basis. These are the only possible translationally
    # invariant operators in this basis.
    totalX = np.array([[op_string.orbital_operators == ['X']] for op_string in basis], dtype=complex)
    totalY = np.array([[op_string.orbital_operators == ['Y']] for op_string in basis], dtype=complex)
    totalZ = np.array([[op_string.orbital_operators == ['Z']] for op_string in basis], dtype=complex)
    
    generators = np.hstack((totalX, totalY, totalZ))

    assert(int(invariant_ops.shape[1]) == 3)
        
    overlaps = np.dot(np.conj(generators.T), invariant_ops)
    
    (eigvals, eigvecs) = nla.eigh(overlaps)
    inds_null_space    = np.where(np.abs(eigvals) < 1e-12)[0]
    
    assert(len(inds_null_space) == 0)
    
def test_kitaev_chain_edge_modes():
    # Check that we find the expected Majorana zero modes
    # for the Kitaev chain.
    
    N  = 12
    mu = 0.5 # mu/t ratio

    # Use a basis of one-site operators.
    basis = qy.cluster_basis(1, np.arange(N), 'Majorana')

    # Expected zero modes
    # \hat{\Psi}_L = \sum_{j=0}^{N-1} (-1)^j (\mu/t)^{2j} \hat{b}_{j}
    # \hat{\Psi}_R = \sum_{j=0}^{N-1} (-1)^j (\mu/t)^{2j} \hat{a}_{N-1-j}
    psi1 = qy.Operator(np.zeros(len(basis)), basis.op_strings)
    psi2 = qy.Operator(np.zeros(len(basis)), basis.op_strings)
    for site in range(N):
        coeff = (-1.0)**site * mu**(2*site)
        psi1 += qy.Operator(np.array([coeff]), [qy.opstring('B {}'.format(site))])
        psi2 += qy.Operator(np.array([coeff]), [qy.opstring('A {}'.format(N-1-site))])
    psi1.normalize()
    psi2.normalize()
        
    expected_ops      = np.zeros((len(basis),2), dtype=complex)
    expected_ops[:,0] = psi1.coeffs
    expected_ops[:,1] = psi2.coeffs

    kitaev_chain = qy.Operator(op_type='Majorana')
    for site1 in range(N):
        kitaev_chain += qy.Operator([-0.5*mu], [qy.opstring('D {}'.format(site1))])
        
        site2 = site1 + 1
        if site2 < N:
            kitaev_chain += qy.Operator([1.0], [qy.opstring('A {} B {}'.format(site1,site2))])

    commuting_ops = qy.commuting_operators(basis, kitaev_chain)
        
    assert(int(commuting_ops.shape[1]) == 2)
    
    overlaps = np.dot(np.conj(expected_ops.T), commuting_ops)
    
    (eigvals, eigvecs) = nla.eigh(overlaps)
    inds_null_space    = np.where(np.abs(eigvals) < 1e-12)[0]

    assert(len(inds_null_space) == 0)

def test_inverse_ssh_model_generation():
    # Check that we find the SSH model
    # when we provide as input its
    # zero modes and other symmetries
    # that it obeys.

    # Number of unit cells
    N  = 12
    # Number of sites
    num_sites = N
    # t'/t ratio
    tp = 2.0 

    # Use a 1D chain lattice with one orbital
    # per unit cell.
    lattice = qy.lattice.chain(N, ['A'], periodic=True)
    
    # Use a distance basis of two-site operators
    # separated up to a unit cell away.
    k = 2
    R = 1.0
    basis = qy.distance_basis(lattice, k, R, 'Majorana')

    # Zero modes of SSH model
    # \hat{\Psi}_1 = \sum_{j=0}^{N/2-1} (-t/t')^j \hat{a}_{2j}
    # \hat{\Psi}_1 = \sum_{j=0}^{N/2-1} (-t/t')^j \hat{b}_{2j}
    # \hat{\Psi}_3 = \sum_{j=0}^{N/2-1} (-t/t')^j \hat{a}_{N-1-2j}
    # \hat{\Psi}_4 = \sum_{j=0}^{N/2-1} (-t/t')^j \hat{b}_{N-1-2j}
    psi1 = qy.Operator(op_type='Majorana')
    psi2 = qy.Operator(op_type='Majorana')
    psi3 = qy.Operator(op_type='Majorana')
    psi4 = qy.Operator(op_type='Majorana')
    for site in range(num_sites//2):
        coeff = (-1.0/tp)**site
        psi1 += qy.Operator(np.array([coeff]), [qy.opstring('A {}'.format(2*site))])
        psi2 += qy.Operator(np.array([coeff]), [qy.opstring('B {}'.format(2*site))])
        psi3 += qy.Operator(np.array([coeff]), [qy.opstring('A {}'.format(N-1-2*site))])
        psi4 += qy.Operator(np.array([coeff]), [qy.opstring('B {}'.format(N-1-2*site))])
    
    # Fermion parity operator
    fermion_parity_op_string = qy.OperatorString(['D']*num_sites, np.arange(num_sites), 'Majorana')
    fermion_parity           = qy.Operator([1.0], [fermion_parity_op_string])
    
    # Total fermion number operator
    totalN = qy.Operator(op_type='Majorana')
    for site in range(num_sites):
        totalN += qy.Operator([-0.5], [qy.opstring('D {}'.format(site))])

    # (Spinless) time-reversal symmetry
    time_reversal = qy.time_reversal()

    # Collect the symmetries.
    symmetries = [fermion_parity, psi1, psi2, psi3, psi4, time_reversal, totalN]

    # Add symmetries to the operator generator.
    op_generator = qy.SymmetricOperatorGenerator(basis)
    for symmetry in symmetries:
        op_generator.add_symmetry(symmetry)

    # Generate operators obeying the given symmetries.
    op_generator.generate()

    # Expected: should return a single operator, the SSH model
    ssh_chain = qy.Operator(np.zeros(len(basis)), basis.op_strings)
    for site1 in range(0,num_sites,2):
        site2 = site1 + 1
        if site2 < num_sites:
            ssh_chain += qy.Operator([-0.5], [qy.opstring('A {} B {}'.format(site1,site2))])
            ssh_chain += qy.Operator([0.5],  [qy.opstring('B {} A {}'.format(site1,site2))])
            
        site3 = site2 + 1
        if site3 < num_sites:
            ssh_chain += qy.Operator([-0.5*tp], [qy.opstring('A {} B {}'.format(site2,site3))])
            ssh_chain += qy.Operator([0.5*tp],  [qy.opstring('B {} A {}'.format(site2,site3))])
    expected_op = ssh_chain

    """
    # For debugging:
    ind_sym = 1
    for ops in op_generator.projected_output_operators:
        #print('ops {} = '.format(ind_sym))
        #qy.print_vectors(basis, ops)

        print('eigvals {} = {}'.format(ind_sym, op_generator.projected_eigenvalues[ind_sym-1]))
        
        ind_sym += 1
    """
    
    # Make sure the shapes of the saved data are consistent.
    for ind_output in range(len(symmetries)):
        # Vectors are all the same sizes
        assert(len(basis) \
               == op_generator.projected_eigenvectors[ind_output].shape[0])
        assert(op_generator.projected_eigenvectors[ind_output].shape[0] \
               == op_generator.eigenvectors[ind_output].shape[0])
        assert(op_generator.eigenvectors[ind_output].shape[0] \
               == op_generator.output_operators[ind_output].shape[0])
        assert(op_generator.output_operators[ind_output].shape[0] \
               == op_generator.projected_output_operators[ind_output].shape[0])

        print(op_generator.projected_superoperators[ind_output].shape)
        print(op_generator.projected_output_operators[ind_output].shape)

        if ind_output > 0:
            # Projected superoperators have same size as the
            # number of projected_output_operators from the previous iteration.
            assert(op_generator.projected_superoperators[ind_output].shape[0] \
                   == op_generator.projected_output_operators[ind_output-1].shape[1])

    ops = op_generator.projected_output_operators[-1]

    assert(int(ops.shape[1]) == 1)

    op = qy.Operator(ops[:,0], basis.op_strings)
    op.normalize(order=np.inf)

    # Due to finite size effects, the final result is not *exactly* the SSH chain,
    # but the SSH chain with an extra bond with small weight connecting the two
    # edges of the chain. I tested that the bond vanishes for larger system
    # sizes, but, to make this test fast, I used a small system size and
    # allowed the tolerance to be large.
    tol = 0.05
    assert((op - expected_op).norm() < tol or (op - (-expected_op)).norm() < tol)
