import qulacs
import openfermion


def parse_of_general_operators(num_qubits, openfermion_operators):
    """convert openfermion operator for generic cases (non-Hermitian operators)
    Args:
        n_qubit (:class:`int`)
        openfermion_op (:class:`openfermion.ops.QubitOperator`)
    Returns:
        :class:`qulacs.GeneralQuantumOperator`
    """
    ret = qulacs.GeneralQuantumOperator(num_qubits)

    for pauli_product in openfermion_operators.terms:
        coef = openfermion_operators.terms[pauli_product]
        pauli_string = ''
        for pauli_operator in pauli_product:
            pauli_string += pauli_operator[1] + ' ' + str(pauli_operator[0])
            pauli_string += ' '
        ret.add_operator(coef, pauli_string[:-1])
    return ret


def qulacs_jordan_wigner(fermion_operator, n_qubits=None):
    """
    wrapper for openfermion.jordan_wigner which directly converts
    openfermion.FermionOperator to qulacs.GeneralQuantumOperator
    Args:
        fermion_operator (openfermion.FermionOperator)
        n_qubits (int):
            # of qubits (if not given, n_qubits is assumed to be
            the number of orbitals which appears in the given fermion operator)
    Return:
        qulacs.GeneralQuantumOperator
    """
    def count_qubit_in_qubit_operator(op):
        n_qubits = 0
        for pauli_product in op.terms:
            for pauli_operator in pauli_product:
                if n_qubits < pauli_operator[0]:
                    n_qubits = pauli_operator[0]
        return n_qubits+1

    qubit_operator = openfermion.transforms.jordan_wigner(fermion_operator)
    _n_qubits = count_qubit_in_qubit_operator(qubit_operator) if n_qubits is None else n_qubits
    qulacs_operator = parse_of_general_operators(_n_qubits, qubit_operator)
    return qulacs_operator
