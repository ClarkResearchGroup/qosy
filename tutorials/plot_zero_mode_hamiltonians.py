#!/usr/bin/env python
"""
This is a script for finding and plotting a
non-interacting BdG Hamiltonian that commutes with
Majorana zero modes with Gaussian distributions in 1D and 2D.

These results are described in more detail in Section III of

E. Chertkov, B. Villalonga, and B. K. Clark, "Engineering Topological Models with a General-Purpose Symmetry-to-Hamiltonian Approach," arXiv:1910.10165 (2019).
"""

import numpy as np
import numpy.linalg as nla

import matplotlib.pyplot as plt
from plot_tools import save_my_fig, plot_2d_zero_mode

import qosy as qy
    
threshold = 1e-12

def find_zero_modes(hamiltonian, num_orbitals, num_vecs=None):
    """Find zero modes that are linear combinations of :math:`\\hat{a}_j`
    and :math:`\\hat{b}_j` operators that commute with the given Hamiltonian.
    
    Parameters
    ----------
    hamiltonian : Operator
       The Hamiltonian to commute with.
    num_orbitals : int
       The number of orbitals in the system.
    num_vecs : int, optional
       If not None, then Lanczos with this many vectors
       is used instead of full diagonalization to find
       the zero modes. Default is None.
    """
    
    basis = qy.Basis()
    for orb in range(num_orbitals):
        basis += qy.opstring('A {}'.format(orb))
        basis += qy.opstring('B {}'.format(orb))
        basis += qy.opstring('D {}'.format(orb))

    gen = qy.SymmetricOperatorGenerator(basis)
    gen.add_symmetry(qy.convert(hamiltonian,'Majorana'),num_vecs=num_vecs)
    gen.generate(verbose=False)

    zero_modes = gen.projected_output_operators[-1]
    #print(gen.projected_eigenvalues[-1])
    
    return zero_modes

def print_info(hamiltonian, zero_modes, expected_zero_modes):
    """Print the given operators for easy comparison.
    """
    
    print('========================')
    print('Hamiltonian:')
    print(qy.convert(hamiltonian,'Fermion'))
    print('Expected zero modes:')
    qy.print_operators(expected_zero_modes)
    
    print('Found zero modes:')
    qy.print_operators(zero_modes)

def two_site_ham1(ai, bi, aj, bj, i, j):
    # Two-site Hamiltonian that commutes with
    #     \alpha_i a_i + \alpha_j a_j
    # and \beta_i b_i + \beta_j b_j

    # Also works
    """
    coeffs    = [ai*bj+aj*bi,
                 ai*bj-aj*bi,
                 -2.0*aj*bj,
                 -2.0*ai*bi]
    """

    coeffs     = [1.0 + (aj/ai)/(bj/bi),
                  1.0 - (aj/ai)/(bj/bi),
                  -2.0*(aj/ai),
                  -2.0*(bi/bj)]

    op_strings = [qy.opstring('CDag {} C {}'.format(i,j)),
                  qy.opstring('CDag {} CDag {}'.format(i,j)),
                  qy.opstring('CDag {} C {}'.format(i,i)),
                  qy.opstring('CDag {} C {}'.format(j,j))]

    norm = (1.0 + (aj/ai)/(bj/bi))

    result = qy.convert(qy.Operator(np.array(coeffs)/norm, op_strings), 'Majorana')
    #print(result)
    
    return result

# ===== 1D Gaussian zero modes example =====

# Specify the parameters of the zero mode.
L = 100
xA = 0.0
xB = L-1.0
sigma = 10.0

x  = np.arange(L)
alphas = np.exp(-((x-xA)/(np.sqrt(2)*sigma))**2.0)
betas  = np.exp(-((x-xB)/(np.sqrt(2)*sigma))**2.0)

t  = 1.0
D  = t*np.tanh((xB-xA)/(2.0*(sigma**2.0)))
mu = np.zeros(L)

# Create the zero mode Hamiltonian from bond operators.
print('D = {}'.format(D))
hamiltonian = qy.Operator([], [], 'Fermion')
for j in range(L-1):
    coeffs     = []
    op_strings = []

    coeffs.append(t)
    op_strings.append(qy.opstring('CDag {} C {}'.format(j, j+1)))
    
    coeffs.append(D)
    op_strings.append(qy.opstring('CDag {} CDag {}'.format(j,j+1)))

    muj   = -2.0*t*np.exp(-(2.0*(x[j]-xA) + 1.0)/(2.0*(sigma**2.0))) / (1.0 + np.exp(-(xB-xA)/(sigma**2.0)))
    mujp1 = -2.0*t*np.exp((2.0*(x[j]-xB) + 1.0)/(2.0*(sigma**2.0))) / (1.0 + np.exp(-(xB-xA)/(sigma**2.0)))
 
    coeffs.append(muj)
    op_strings.append(qy.opstring('CDag {} C {}'.format(j,j)))

    coeffs.append(mujp1)
    op_strings.append(qy.opstring('CDag {} C {}'.format(j+1,j+1)))

    mu[j]   += muj
    mu[j+1] += mujp1
    
    h_ij = qy.Operator(coeffs, op_strings, 'Fermion')
    
    hamiltonian += h_ij
    
"""
t_i = np.zeros(num_orbitals)
D_i = np.zeros(num_orbitals)
mu_i = np.zeros(num_orbitals)
for (coeff, os) in qy.convert(hamiltonian, 'Fermion'):
    if os.orbital_operators[0] == 'CDag' and os.orbital_operators[1] == 'CDag':
        D_i[os.orbital_labels[0]] = np.real(coeff)
    else:
        if os.orbital_labels[0] == os.orbital_labels[1]:
            mu_i[os.orbital_labels[0]] = np.real(coeff)
        else:
            t_i[os.orbital_labels[0]] = np.real(coeff)
"""   

# Check with ED (for small systems) if the spectrum
# is doubly-degenerate as it should be if these
# zero mode symmetries exist.
if L <= 10:
    (evals, evecs) = qy.diagonalize(hamiltonian, L)
    
    check = True
    for ind_eval in range(0,len(evals),2):
        if not np.isclose(evals[ind_eval], evals[ind_eval+1]):
            check=False
            break
    print('All eigenvalues are doubly degenerate: {}'.format(check))

    plt.figure()
    plt.plot(evals)
    plt.ylabel('Many-body eigenvalue')

"""
# This is slow, but can be uncommented out and ran to
# inspect the Hamiltonian's spectrum.

(gs_energy, evals_onebody, evecs_onebody, evecs_onebody_majorana) = qy.diagonalize_quadratic(hamiltonian, L)
inds_zm = np.where(np.abs(evals_onebody) < 1e-10)[0]
print('Number of zero modes: {}'.format(len(inds_zm)))
ind_zm  = inds_zm[0]
plt.figure()
plt.plot(np.sort(evals_onebody))
plt.ylabel('Single particle energies')
plt.figure()
plt.plot(evecs_onebody[:,ind_zm])
plt.ylabel('Single particle eigenstate')
"""

"""
Print out the Hamiltonian.
zero_modes = find_zero_modes(hamiltonian, L)
print(':::::::::::::::::')
print('Test Hamiltonian:')
qy.print_operators([hamiltonian], convert_to='Fermion')
print('Zero modes:')
qy.print_operators(zero_modes, norm_order=np.inf)
"""

# Plot the \alpha_i and \beta_j of the 1D gaussian zero modes.
plt.figure()
plt.plot(alphas/nla.norm(alphas), 'b-', label='$\\alpha_{x}$')
plt.plot(betas/nla.norm(betas), 'g--', label='$\\beta_{x}$')
plt.xlabel('$x$')
plt.legend()

# Save the figure to a file.
save_my_fig('gaussian_zero_modes_1d.pdf')

# Plot the chemical potentials of the 1D gaussian zero mode Hamiltonian.
plt.figure()
#plt.plot(D*np.ones(L), 'k-', markersize=10, markeredgecolor='k', linewidth=3, label='$\\Delta_{\\mathbf{x}}/t$')
plt.plot(mu, 'r-', linewidth=3, markersize=10, markeredgecolor='r')
plt.xlabel('$x$')
plt.ylabel('$\\mu_{x}/t$')

# Save the figure to a file.
save_my_fig('gaussian_zero_modes_hamiltonian_1d.pdf')

# ===== 2D Gaussian zero modes example =====

# The parameters for the 2D Gaussian zero modes.
L1 = 100
L2 = 100
xA = np.array([L1/4.0, L2/4.0])
xB = np.array([3.0/4.0*L1, 3.0/4.0*L2])
sigma = 10.0
t  = 1.0

a1 = np.array([1.0, 0.0])
a2 = np.array([0.0, 1.0])

(Xs, Ys) = np.meshgrid(np.arange(L1), np.arange(L2))

alphas = np.zeros((L1,L2))
betas  = np.zeros((L1,L2))
mus    = np.zeros((L1,L2))
D1     = t*np.tanh(np.dot(xB-xA, a1)/(2.0*(sigma**2.0)))
D2     = t*np.tanh(np.dot(xB-xA, a2)/(2.0*(sigma**2.0)))

print('D1 = {}\nD2 = {}'.format(D1,D2))

# Create the zero mode Hamiltonian from bond operators.
for ind1 in range(L1):
    for ind2 in range(L2):
        x = ind1*a1 + ind2*a2
        alphas[ind1,ind2] = np.exp(-(nla.norm(x-xA)/(np.sqrt(2)*sigma))**2.0)
        betas[ind1,ind2]  = np.exp(-(nla.norm(x-xB)/(np.sqrt(2)*sigma))**2.0)

        if ind1 < L1-1:
            muj1 = -2.0*t*np.exp(-(2.0*np.dot(x-xA, a1) + 1.0)/(2.0*(sigma**2.0))) / (1.0 + np.exp(-np.dot(xB-xA,a1)/(sigma**2.0)))
            mujp1 = -2.0*t*np.exp((2.0*np.dot(x-xB,a1) + 1.0)/(2.0*(sigma**2.0))) / (1.0 + np.exp(-np.dot(xB-xA,a1)/(sigma**2.0)))

            mus[ind1,ind2]   += muj1
            mus[ind1+1,ind2] += mujp1
            
        if ind2 < L2-1:
            muj2 = -2.0*t*np.exp(-(2.0*np.dot(x-xA, a2) + 1.0)/(2.0*(sigma**2.0))) / (1.0 + np.exp(-np.dot(xB-xA,a2)/(sigma**2.0)))
            mujp2 = -2.0*t*np.exp((2.0*np.dot(x-xB,a2) + 1.0)/(2.0*(sigma**2.0))) / (1.0 + np.exp(-np.dot(xB-xA,a2)/(sigma**2.0)))

            mus[ind1,ind2]   += muj2
            mus[ind1,ind2+1] += mujp2
        
coeffs     = []
op_strings = []
for ind1 in range(L1):
    for ind2 in range(L2):
        orb  = ind2*L1 + ind1
        
        if ind1 < L1-1:
            orb1 = ind2*L1 + (ind1+1) 
            
            coeffs.append(t)
            op_strings.append(qy.opstring('CDag {} C {}'.format(orb, orb1)))
            
            coeffs.append(D1)
            op_strings.append(qy.opstring('CDag {} CDag {}'.format(orb,orb1)))
            
        if ind2 < L2-1:
            orb2 = (ind2+1)*L1 + ind1
            
            coeffs.append(t)
            op_strings.append(qy.opstring('CDag {} C {}'.format(orb, orb2)))

            coeffs.append(D2)
            op_strings.append(qy.opstring('CDag {} CDag {}'.format(orb,orb2)))
            
        coeffs.append(mus[ind1,ind2])
        op_strings.append(qy.opstring('CDag {} C {}'.format(orb,orb)))

hamiltonian = qy.Operator(coeffs, op_strings, 'Fermion')

if L <= 10:
    (evals, evecs) = qy.diagonalize(hamiltonian, L)
    
    check = True
    for ind_eval in range(0,len(evals),2):
        if not np.isclose(evals[ind_eval], evals[ind_eval+1]):
            check=False
            break
    print('All eigenvalues are doubly degenerate: {}'.format(check))

    plt.figure()
    plt.plot(evals)
    plt.ylabel('Many-body eigenvalue')

# Check visually that the zero modes are correct
"""
zero_modes = find_zero_modes(hamiltonian, L, num_vecs=4)
plot_2d_zero_mode(zero_modes[0], L1, L2)
plt.suptitle('Real zero mode 1')
plot_2d_zero_mode(zero_modes[1], L1, L2)
plt.suptitle('Real zero mode 2')
"""

#print(':::::::::::::::::')
#print('Test Hamiltonian:')
#qy.print_operators([hamiltonian], convert_to='Fermion')
#print('Zero modes:')
#qy.print_operators(zero_modes, norm_order=np.inf)

"""
plt.figure()
plt.imshow(alphas/nla.norm(alphas), origin='lower', cmap=plt.get_cmap('Blues'))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar()
plt.title('$\\alpha_{\\mathbf{x}}$')

save_my_fig('gaussian_zero_mode1_2d.pdf')

plt.figure()
plt.imshow(betas/nla.norm(betas), origin='lower', cmap=plt.get_cmap('Greens'))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar()
plt.title('$\\beta_{\\mathbf{x}}$')

save_my_fig('gaussian_zero_mode2_2d.pdf')
"""

# Plot the 2D Gaussian zero mode's \alpha_i and \beta_j parameters.
plot_2d_zero_mode((alphas, betas), L1, L2, origin='lower')
save_my_fig('gaussian_zero_modes_2d.pdf')

plt.figure()

# Plot the complex-valued pairings \Delta_{i,j}
# as vectors whose angles (and colors) are the phase of the complex
# number and whose length are the amplitude of the complex number.
Dangle = 1.0/8.0*np.ones(Xs.shape) # (pi/4)/(2pi) = 1/8
plt.quiver(Xs, Ys, D1*np.ones(Xs.shape), D2*np.ones(Ys.shape), Dangle, cmap=plt.get_cmap('hsv'))
plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.colorbar()
plt.clim(-1.0, 1.0)
plt.title('$\\Delta_{\\mathbf{x}}/t$')

# Save the figure to a file.
save_my_fig('gaussian_zero_modes_pairing_2d.pdf')

plt.figure()

# Plot the 2D Gaussian zero mode Hamiltonian's chemical potentials.
plt.imshow(mus, origin='lower', cmap=plt.get_cmap('afmhot'))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xticks([0,L1//2,L1])
plt.yticks([0,L2//2,L2])
plt.colorbar()
plt.title('$\\mu_{\\mathbf{x}}/t$')

# Save the figure to a file.
save_my_fig('gaussian_zero_modes_chemical_potential_2d.pdf')

plt.show()

