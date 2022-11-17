import itertools
from openfermion import up_index, down_index
from openfermion.ops import FermionOperator

from qulacs import ParametricQuantumCircuit

from vqe_qmmm.vqe.openfermion_qulacs import qulacs_jordan_wigner


def add_parametric_multi_Pauli_rotation_gate(circuit, indices,
                                             pauli_ids, theta,
                                             parameter_ref_index=None,
                                             parameter_coef=1.0):
    circuit.add_parametric_multi_Pauli_rotation_gate(indices,
                                                     pauli_ids, theta)
    return circuit


def add_parametric_circuit_using_generator(circuit,
                                           generator, theta,
                                           param_index, coef=1.0):
    for i_term in range(generator.get_term_count()):
        pauli = generator.get_term(i_term)
        pauli_id_list = pauli.get_pauli_id_list()
        pauli_index_list = pauli.get_index_list()
        pauli_coef = pauli.get_coef().imag #coef should be pure imaginary
        circuit = add_parametric_multi_Pauli_rotation_gate(circuit,
                        pauli_index_list, pauli_id_list,
                        theta, parameter_ref_index=param_index,
                        parameter_coef=coef*pauli_coef)
    return circuit


def add_theta_value_offset(theta_offsets, generator, ioff):
    pauli_coef_lists = []
    for i in range(generator.get_term_count()):
        pauli = generator.get_term(i)
        pauli_coef_lists.append(pauli.get_coef().imag) #coef should be pure imaginary
    theta_offsets.append([generator.get_term_count(), ioff,
                          pauli_coef_lists])
    ioff = ioff + generator.get_term_count()
    return theta_offsets, ioff




class UCCSD1():

    def __init__(self, n_qubit, n_electron):
        self.n_electron = n_electron
        self.n_qubit = n_qubit
        self.circuit = ParametricQuantumCircuit(self.n_qubit)
        self.theta_offsets = []
        self.params = []

        nocc = int(n_electron/2)
        nvirt = int((n_qubit-n_electron)/2)

        # Build the circuit
        ioff = 0
        # Singles
        spin_index_functions = [up_index, down_index]
        for i_t1, (a, i) in enumerate(
            itertools.product(range(nvirt), range(nocc))):
            a_spatial = a + nocc
            i_spatial = i
            for ispin in range(2):
                #Spatial Orbital Indices
                so_index = spin_index_functions[ispin]
                a_spin_orbital = so_index(a_spatial)
                i_spin_orbital = so_index(i_spatial)
                #t1 operator
                qulacs_generator = self._generate_t1(a_spin_orbital,
                                                i_spin_orbital)
                #Add t1 into the circuit
                theta = 0.0
                self.params.append(theta)
                self.theta_offsets, ioff = add_theta_value_offset(self.theta_offsets,
                                                                  qulacs_generator,
                                                                  ioff)
                self.circuit = add_parametric_circuit_using_generator(self.circuit,
                                                                      qulacs_generator,
                                                                      theta,i_t1,1.0)

        # Dobules (alpha, alpha, beta, beta)
        for i_t2, (a, i, b, j) in enumerate(
                itertools.product(range(nvirt), range(nocc),
                                  range(nvirt), range(nocc))):

                a_spatial = a + nocc
                i_spatial = i
                b_spatial = b + nocc
                j_spatial = j

                #Spatial Orbital Indices
                aa = up_index(a_spatial)
                ia = up_index(i_spatial)
                bb = down_index(b_spatial)
                jb = down_index(j_spatial)
                #t1 operator
                qulacs_generator = self._generate_t2(aa, ia,
                                                bb, jb)
                #Add p-t2 into the circuit
                theta = 0.0
                self.params.append(theta)
                self.theta_offsets, ioff = add_theta_value_offset(self.theta_offsets,
                                                                  qulacs_generator,
                                                                  ioff)
                self.circuit = add_parametric_circuit_using_generator(self.circuit,
                                                                      qulacs_generator,
                                                                      theta,i_t2,1.0)

        # Dobules (beta, beta, beta, beta)
        for i_t2, (a, i, b, j) in enumerate(
                itertools.product(range(nvirt), range(nocc),
                                  range(nvirt), range(nocc))):

                a_spatial = a + nocc
                i_spatial = i
                b_spatial = b + nocc
                j_spatial = j

                #Spatial Orbital Indices
                ab = down_index(a_spatial)
                ib = down_index(i_spatial)
                bb = down_index(b_spatial)
                jb = down_index(j_spatial)
                #t1 operator
                qulacs_generator = self._generate_t2(ab, ib,
                                                bb, jb)
                #Add p-t2 into the circuit
                theta = 0.0
                self.params.append(theta)
                self.theta_offsets, ioff = add_theta_value_offset(self.theta_offsets,
                                                                  qulacs_generator,
                                                                  ioff)
                self.circuit = add_parametric_circuit_using_generator(self.circuit,
                                                                      qulacs_generator,
                                                                      theta,i_t2,1.0)

        # Dobules (alpha, alpha, alpha, alpha)
        for i_t2, (a, i, b, j) in enumerate(
                itertools.product(range(nvirt), range(nocc),
                                  range(nvirt), range(nocc))):

                a_spatial = a + nocc
                i_spatial = i
                b_spatial = b + nocc
                j_spatial = j

                #Spatial Orbital Indices
                aa = up_index(a_spatial)
                ia = up_index(i_spatial)
                ba = up_index(b_spatial)
                ja = up_index(j_spatial)
                #t1 operator
                qulacs_generator = self._generate_t2(aa, ia,
                                                ba, ja)
                #Add p-t2 into the circuit
                theta = 0.0
                self.params.append(theta)
                self.theta_offsets, ioff = add_theta_value_offset(self.theta_offsets,
                                                                  qulacs_generator,
                                                                  ioff)
                self.circuit = add_parametric_circuit_using_generator(self.circuit,
                                                                      qulacs_generator,
                                                                      theta,i_t2,1.0)

#=============================
    def update_circuit_param(self, theta_list):
        for idx, theta in enumerate(theta_list):
            for ioff in range(self.theta_offsets[idx][0]):
                pauli_coef = self.theta_offsets[idx][2][ioff]
                self.circuit.set_parameter(self.theta_offsets[idx][1]+ioff,
                                      theta*pauli_coef) #量子回路にパラメータをセット
        self.params = theta_list

#=============================
    def _generate_t1(self, a, i):
        # Generate single excitations
        generator = FermionOperator((
                (a, 1),
                (i, 0)),
                1.0)
        generator += FermionOperator((
                (i, 1),
                (a, 0)),
                -1.0)
        #JW-transformation of c^\dagger_a a_i - c^\dagger_i a_a
        qulacs_generator = qulacs_jordan_wigner(generator)
        return qulacs_generator
#=============================
    def _generate_t2(self, a, i, b, j):
        # Generate double excitations
        generator = FermionOperator((
                (a, 1),
                (b, 1),
                (j, 0),
                (i, 0)),
                1.0)
        generator += FermionOperator((
                (i, 1),
                (j, 1),
                (b, 0),
                (a, 0)),
                -1.0)
        #JW-transformation of c^\dagger_a c^\dagger_b c_j c_i - c^\dagger_i c^\dagger_j c_b c_a
        qulacs_generator = qulacs_jordan_wigner(generator)
        return qulacs_generator
#=============================
    def get_parameter_count(self):
        """get_parameter_count
        returns # of parameter
        """
        return len(self.params)
