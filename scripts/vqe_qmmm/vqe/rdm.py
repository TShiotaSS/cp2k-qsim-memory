import numpy as np
from itertools import product

from qulacs import Observable

from openfermion.ops import FermionOperator

from vqe_qmmm.vqe.openfermion_qulacs import qulacs_jordan_wigner


def get_1rdm(state, fermion_qubit_mapping=qulacs_jordan_wigner, n_spin_orbitals=None):
    """
    compute 1rdm of a given state with user specfied fermion to qubit mapping.

    Args:
        state (qulacs.QuantumState)
        fermion_qubit_mapping (func):
            function for mapping openfermion.FermionOperator to qulacs.GeneralQuantumOperator.
            its argument should be (FermionOperator)
        n_spin_orbitals (int):
            number of spin-orbitals. if None, it is set to # of qubits in QuantumState

    Return:
        numpy.ndarray of shape (n_spin_orbitals, n_spin_orbitals):
            1-RDM
    """
    _n_spin_orbitals = n_spin_orbitals if n_spin_orbitals is not None else state.get_qubit_count()
    ret = np.zeros((_n_spin_orbitals, _n_spin_orbitals), dtype=np.complex128)
    for i in range(_n_spin_orbitals):
        for j in range(i+1):
            one_body_op = FermionOperator(((i, 1), (j, 0)))
            one_body_op = fermion_qubit_mapping(one_body_op)
            tmp = one_body_op.get_expectation_value(state)
            ret[i,j] = tmp
            ret[j,i] = tmp.conjugate()
    return ret


def get_2rdm(state, fermion_qubit_mapping=qulacs_jordan_wigner, n_spin_orbitals=None):
    """
    compute 2rdm of a given state with user specfied fermion to qubit mapping.
    Args:
        state (qulacs.QuantumState)
        fermion_qubit_mapping (func):
            function for mapping openfermion.FermionOperator to qulacs.GeneralQuantumOperator.
            its argument should be (FermionOperator)
        n_spin_orbitals (int):
            number of spin-orbitals. if None, it is set to # of qubits in QuantumState
    Return:
        numpy.ndarray of shape (n_spin_orbitals, n_spin_orbitals, n_spin_orbitals, n_spin_orbitals):
            2-RDM
    """
    _n_spin_orbitals = n_spin_orbitals if n_spin_orbitals is not None else state.get_qubit_count()
    ret = np.zeros((_n_spin_orbitals, _n_spin_orbitals, _n_spin_orbitals, _n_spin_orbitals), dtype=np.complex128)
    for i, k in product(range(_n_spin_orbitals), range(_n_spin_orbitals)):
        for j, l in product(range(i), range(k)):
            two_body_op = FermionOperator(((i,1),(j,1),(k,0),(l,0)))
            two_body_op = fermion_qubit_mapping(two_body_op)
            tmp = two_body_op.get_expectation_value(state)
            ret[i,j,k,l] = tmp
            ret[i,j,l,k] = -tmp
            ret[j,i,l,k] = tmp
            ret[j,i,k,l] = -tmp
            ret[l,k,j,i] = tmp.conjugate()
            ret[k,l,j,i] = -tmp.conjugate()
            ret[k,l,i,j] = tmp.conjugate()
            ret[l,k,i,j] = -tmp.conjugate()

    return ret
