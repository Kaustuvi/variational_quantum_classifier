from .data_preprocessor import DataPreprocessor
from math import log2
from cirq import Circuit, InsertStrategy, GridQubit, Ry, CNOT, NamedQubit, LineQubit
from sympy.combinatorics.graycode import GrayCode
import numpy as np


class StatePreparation:
    def __init__(self, preprocessed_data, number_of_qubits):
        self.preprocessed_data = preprocessed_data
        self.number_of_qubits = number_of_qubits
        self.qubits = [LineQubit(i) for i in range(self.number_of_qubits)]
        self.rotation_angles = None
        self.state_preparation_circuit = None

    def get_angles_for_state_preparation(self):
        self.rotation_angles = [self.get_angles_from_data(
            data, self.number_of_qubits) for data in self.preprocessed_data]
        return self.rotation_angles

    def generate_state_preparation_circuit(self, rotation_angles_list):
        state_preparation_circuit = Circuit()
        for rotation_angles in rotation_angles_list:
            for angles in rotation_angles:
                qubit_index = rotation_angles.index(angles)
                if(qubit_index == 0):
                    qubit_angle = angles[qubit_index]
                    RY = Ry(qubit_angle)
                    state_preparation_circuit.append(
                        [RY(self.qubits[qubit_index])], strategy=InsertStrategy.EARLIEST)
                else:
                    gray_code_list = generate_gray_code(
                        qubit_index)
                    for qubit_angle in angles:
                        RY = Ry(qubit_angle)
                        state_preparation_circuit.append(
                            [RY(self.qubits[qubit_index])], strategy=InsertStrategy.EARLIEST)
                        l = angles.index(qubit_angle)
                        cnot_position = self.find_cnot_position(
                            gray_code_list[(l+1) % len(angles)], gray_code_list[l % len(angles)])
                        state_preparation_circuit += self.get_cnot_circuit(
                            self.qubits[cnot_position[0]], self.qubits[qubit_index])
        self.state_preparation_circuit = state_preparation_circuit
        return self.state_preparation_circuit

    @staticmethod
    def get_angles_from_data(preprocessed_data, number_of_qubits):
        rotation_angles = []
        for k in range(number_of_qubits):
            alpha_jk = []
            for j in range(2**(number_of_qubits-k-1)):
                alpha_numerator = 0
                alpha_denominator = 0
                for l in range(2**k):
                    numerator_index = (2*j+1)*2**k+l
                    alpha_numerator += preprocessed_data[numerator_index]**2
                for l in range(2**(k+1)):
                    denominator_index = j*2**(k+1)+l
                    alpha_denominator += preprocessed_data[denominator_index]**2
                alpha_jk.append(
                    2 * np.arcsin(np.sqrt(alpha_numerator) / np.sqrt(alpha_denominator)))
            M = get_multiplication_matrix(j+1, log2(j+1))
            rotation_angles.insert(0, list(np.matmul(M, alpha_jk)))
        return rotation_angles

    @staticmethod
    def get_cnot_circuit(control_line, target_line):
        cnot_circuit = Circuit()
        cnot_circuit.append([CNOT(control_line, target_line)],
                            strategy=InsertStrategy.EARLIEST)
        return cnot_circuit

    @staticmethod
    def find_cnot_position(curr_gray_code, prev_gray_code):
        return [i for i in range(len(curr_gray_code)) if curr_gray_code[i] != prev_gray_code[i]]

def get_multiplication_matrix(size, number_of_controls):
    M = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            binary_j = format(j, '0'+str(int(log2(size)))+'b')
            gray_i = format(i ^ (i >> 1), '0'+str(int(log2(size)))+'b')
            bitwise_product = 0
            for index in range(len(binary_j)):
                bitwise_product += int(binary_j[index])*int(gray_i[index])
            M[i][j] = (-1)**bitwise_product
    M = 2**(-number_of_controls)*np.array(M)
    return M

def generate_gray_code(number_of_controls):
    return list(GrayCode(number_of_controls).generate_gray())
