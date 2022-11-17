import numpy as np
from itertools import product

from scipy.optimize import minimize
from pyscf import ao2mo

from qulacs import QuantumState

from openfermion.ops import InteractionOperator, FermionOperator
from openfermion.transforms import get_fermion_operator

from vqe_qmmm.vqe.openfermion_qulacs import qulacs_jordan_wigner
from vqe_qmmm.vqe.rdm import get_1rdm, get_2rdm
from vqe_qmmm.vqe.uccsd1 import UCCSD1

class VQECI(object):

    def __init__(self, mol=None,
                 fermion_qubit_mapping=qulacs_jordan_wigner):
        self.mol = mol
        self.ansatz = None # to be used to store the ansatz
        self.fermion_qubit_mapping = fermion_qubit_mapping
        self.opt_param = None # to be used to store the optimal parameter for the VQE

#======================
    def get_active_hamiltonian(self, h1, h2, norb, nelec, ecore):
        n_orbitals = h1.shape[0]
        n_qubits = 2 * n_orbitals

        self.n_orbitals = n_orbitals
        self.n_qubit = n_qubits
        self.n_electron = nelec[0] + nelec[1]

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = np.zeros((n_qubits, n_qubits))
        two_body_coefficients = np.zeros((n_qubits, n_qubits,
                                             n_qubits, n_qubits))
        # Set MO one and two electron-integrals
        # according to OpenFermion conventions
        one_body_integrals = h1
        h2_ = ao2mo.restore(1, h2.copy(), n_orbitals) # no permutation see two_body_integrals of _pyscf_molecular_data.py
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
        return active_hamiltonian

#=======================================================================================
    def kernel(self, h1, h2, norb, nelec, ecore=0, **kwargs):
        # Get the active space Hamiltonian
        active_hamiltonian = self.get_active_hamiltonian(h1, h2, norb, nelec, ecore)
        # Convert the Hamiltonian using Jordan Wigner
        fermionic_hamiltonian = get_fermion_operator(active_hamiltonian)
        self.qulacs_hamiltonian = self.fermion_qubit_mapping(fermionic_hamiltonian,
                                                             self.n_qubit)
        # Set initial Quantum State
        self.initial_state = QuantumState(self.n_qubit)
        self.initial_state.set_computational_basis(
            int('0b'+'0'*(self.n_qubit - self.n_electron)+'1'*(self.n_electron),2))
        # ansatz and Initial parameters for VQE
        self.ansatz = UCCSD1(self.n_qubit, self.n_electron)
        init_theta_list = np.zeros(
            self.ansatz.get_parameter_count()) if self.opt_param is None else self.opt_param
        # VQE cost function
        def cost(param):
            state = self.initial_state.copy()
            self.ansatz.update_circuit_param(param)
            self.ansatz.circuit.update_quantum_state(state)
            cost = self.qulacs_hamiltonian.get_expectation_value(state).real
            print(cost)
            return cost 

        # Set optimizer for VQE
        print("----VQE-----")
        method = "BFGS"
        options = {"disp": True, "maxiter": 200, "gtol": 1e-5}
        #perform VQE
        opt = minimize(cost, init_theta_list, method = method, options = options)
        self.opt_param = opt.x
        #print(opt.x)

        # store optimal state
        self.opt_state = self.initial_state.copy()
        self.ansatz.update_circuit_param(self.opt_param)
        self.ansatz.circuit.update_quantum_state(self.opt_state)

        # Get energy
        e = opt.fun.real
        return e, None
#======================
    def make_rdm1(self, state, norb, nelec, link_index=None, **kwargs):
        dm1 = self.one_rdm()
        return dm1
#======================
    def make_rdm12(self, state, norb, nelec, link_index=None, **kwargs):
        dm2 = self.two_rdm()
        return self.make_rdm1(state,norb,nelec), dm2
#======================
    def spin_square(self, civec, norb, nelec):
        return 0, 1
#======================
    def one_rdm(self):
        vqe_one_rdm = np.zeros((self.n_orbitals,self.n_orbitals))
        # get 1 rdm
        spin_dependent_rdm = np.real(get_1rdm(self.opt_state,
                 lambda x: self.fermion_qubit_mapping(x, n_qubits=self.n_qubit)))
        # transform it to spatial rdm
        vqe_one_rdm += spin_dependent_rdm[::2,::2] + spin_dependent_rdm[1::2,1::2]
        self.my_one_rdm = vqe_one_rdm
        return vqe_one_rdm
#======================
    def dm2_elem(self,i,j,k,l):
        jw_hamiltonian = self.fermion_qubit_mapping(FermionOperator(
            ((i, 1), (j, 1), (k, 0), (l, 0))), self.n_qubit)
        two_rdm_real = jw_hamiltonian.get_expectation_value(self.opt_state).real
        #
        # pyscf use real spin-free RDM (i.e. RDM in spatial orbitals)
        #
        return two_rdm_real
#======================
    def two_rdm(self):
        vqe_two_rdm = np.zeros((self.n_orbitals,self.n_orbitals,
                                self.n_orbitals,self.n_orbitals))
        dm2aa = np.zeros_like(vqe_two_rdm)
        dm2ab = np.zeros_like(vqe_two_rdm)
        dm2ba = np.zeros_like(vqe_two_rdm)
        dm2bb = np.zeros_like(vqe_two_rdm)

        # generate 2 rdm
        spin_dependent_rdm = np.real(get_2rdm(self.opt_state,
                   lambda x: self.fermion_qubit_mapping(x, n_qubits=self.n_qubit)))

        # convert it into spatial
        n_orbital = int(self.n_qubit/2)
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
        self.my_two_rdm =(dm2aa.transpose(0,3,1,2) + dm2bb.transpose(0,3,1,2)
                          + dm2ab.transpose(0,3,1,2) 
                          + (dm2ab.transpose(0,3,1,2)).transpose(2, 3, 0, 1))
        return self.my_two_rdm
#======================
    def make_dm2(self):
        dm2 = np.zeros((self.n_orbitals,self.n_orbitals,
                        self.n_orbitals,self.n_orbitals))
        n_orbital = int(self.n_qubit/2)
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
            dm2[i,j,k,l] = (self.dm2_elem(ia,ja,ka,la)
                            + self.dm2_elem(ib,jb,kb,lb)
                            + self.dm2_elem(ia,ja,kb,lb)
                            + self.dm2_elem(ib,jb,ka,la)
            )
        return dm2
#======================
