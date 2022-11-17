import sys
import numpy as np
from scipy.optimize import minimize
from itertools import product

from fcidump_helpers import read_fcidump, write_fcidump, calc_energy, get_hf_rdms, calc_reference_energy
from openfermion.ops import InteractionOperator, FermionOperator
from openfermion.transforms import get_fermion_operator

from qulacs import QuantumState

from vqe_qmmm.vqe import vqemcscf
from vqe_qmmm.util import utils
from vqe_qmmm.vqe.rdm import get_1rdm, get_2rdm
from vqe_qmmm.vqe.openfermion_qulacs import qulacs_jordan_wigner
from vqe_qmmm.vqe.uccsd1 import UCCSD1

def qualcs_perform_davidson(filename: str, output: str) -> None:
    ecore, h, eri, norb, nelec, ms2, uhf = read_fcidump(filename)

    #def get_active_hamiltonian(self, h1, h2, norb, nelec, ecore):
    if True:
        n_orbitals = norb 
        n_qubits = 2 * n_orbitals

        n_orbitals = n_orbitals
        n_qubit = n_qubits
        n_electron = nelec

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = np.zeros((n_qubits, n_qubits))
        two_body_coefficients = np.zeros((n_qubits, n_qubits,
                                             n_qubits, n_qubits))
        # Set MO one and two electron-integrals
        # according to OpenFermion conventions
        one_body_integrals = h[0]
        h2_ = eri[0] #ao2mo.restore(1, h2.copy(), n_orbitals) # no permutation see two_body_integrals of _pyscf_molecular_data.py
        assert h2_.shape[0] == n_orbitals
        two_body_integrals = np.asarray(
            h2_.transpose(0, 2, 3, 1), order='C')

        ############################### Taken from OpenFermion
        # Loop through integrals.
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):

                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[
                    p, q]
                one_body_coefficients[2 * p + 1, 2 *
                                      q + 1] = one_body_integrals[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):

                        # Mixed spin
                        two_body_coefficients[2 * p, 2 * q + 1,
                                              2 * r + 1, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.)
                        two_body_coefficients[2 * p + 1, 2 * q,
                                              2 * r, 2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.)

                        # Same spin
                        two_body_coefficients[2 * p, 2 * q,
                                              2 * r, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.)
                        two_body_coefficients[2 * p + 1, 2 * q + 1,
                                              2 * r + 1, 2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.)

        # Get Hamiltonian in OpenFermion format
        active_hamiltonian = InteractionOperator(
        ecore, one_body_coefficients, two_body_coefficients)

        fermionic_hamiltonian = get_fermion_operator(active_hamiltonian)
        qulacs_hamiltonian = qulacs_jordan_wigner(fermionic_hamiltonian,
                                                  n_qubit)
        # Set initial Quantum State
        initial_state = QuantumState(n_qubit)
        initial_state.set_computational_basis(
            int('0b'+'0'*(n_qubit - n_electron)+'1'*(n_electron),2))
        # ansatz and Initial parameters for VQE
        ansatz = UCCSD1(n_qubit, n_electron)

        opt_param = None
        init_theta_list = np.zeros(ansatz.get_parameter_count())
        # VQE cost function
        def cost(param):
            state = initial_state.copy()
            ansatz.update_circuit_param(param)
            ansatz.circuit.update_quantum_state(state)
            return qulacs_hamiltonian.get_expectation_value(state).real

        # Set optimizer for VQE
        print("----VQE-----")
        method = "BFGS"
        options = {"disp": True, "maxiter": 200, "gtol": 1e-5}
        #perform VQE
        opt = minimize(cost, init_theta_list, method = method, options = options)
        opt_param = opt.x
        #print(opt.x)

        # store optimal state
        opt_state = initial_state.copy()
        ansatz.update_circuit_param(opt_param)
        ansatz.circuit.update_quantum_state(opt_state)

        # Get energy
        e = opt.fun.real

        vqe_one_rdm = np.zeros((n_orbitals, n_orbitals))
        # get 1 rdm
        spin_dependent_rdm = np.real(get_1rdm(opt_state,
                 lambda x: qulacs_jordan_wigner(x, n_qubits=n_qubit)))
        # transform it to spatial rdm
        vqe_one_rdm += spin_dependent_rdm[::2,::2] + spin_dependent_rdm[1::2,1::2]
        my_one_rdm = vqe_one_rdm
        print(my_one_rdm)

        vqe_two_rdm = np.zeros((n_orbitals, n_orbitals,
                                n_orbitals, n_orbitals))
        dm2aa = np.zeros_like(vqe_two_rdm)
        dm2ab = np.zeros_like(vqe_two_rdm)
        dm2ba = np.zeros_like(vqe_two_rdm)
        dm2bb = np.zeros_like(vqe_two_rdm)

        # generate 2 rdm
        spin_dependent_rdm = np.real(get_2rdm(opt_state,
                   lambda x: qulacs_jordan_wigner(x, n_qubits=n_qubit)))

        # convert it into spatial
        n_orbital = int(n_qubit/2)
        for i, j, k, l in product(range(n_orbital),range(n_orbital),
                                  range(n_orbital),range(n_orbital)):
            ia = 2*i
            ja = 2*j
            ka = 2*k
            la = 2*l
            ib = 2*i + 1
            jb = 2*j + 1
            kb = 2*k + 1
            lb = 2*l + 1
            # aa
            dm2aa[i,j,k,l] = spin_dependent_rdm[ia,ja,ka,la]
            # bb
            dm2bb[i,j,k,l] = spin_dependent_rdm[ib,jb,kb,lb]
            #
            dm2ab[i,j,k,l] = spin_dependent_rdm[ia,jb,kb,la]
        my_two_rdm =(dm2aa.transpose(0,3,1,2) + dm2bb.transpose(0,3,1,2)
                          + dm2ab.transpose(0,3,1,2) 
                          + (dm2ab.transpose(0,3,1,2)).transpose(2, 3, 0, 1))

        one_rdm_check = np.zeros_like(my_one_rdm)
        for i in range(n_orbitals):
            one_rdm_check[:, :] += my_two_rdm[:, :, i, i]
        assert np.allclose(one_rdm_check, my_one_rdm * (n_electron - 1))
        print(my_two_rdm)

        reference_rdms = get_hf_rdms(nelec, ms2, norb, uhf)
        reference_energy = calc_energy(ecore, h, eri, *reference_rdms)

        e -= reference_energy
        my_one_rdm -= reference_rdms[0]
        my_two_rdm -= reference_rdms[1]
        write_fcidump(output, e, my_one_rdm, my_two_rdm, n_orbital, n_electron, ms2, uhf)


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 3:
        print("expecting exactly two arguments (input/output file names)")
        sys.exit(5)

    qualcs_perform_davidson(arg[1], arg[2])
