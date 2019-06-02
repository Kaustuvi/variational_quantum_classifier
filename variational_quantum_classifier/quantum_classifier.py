from .data_preprocessor import DataPreprocessor
from math import log2
import numpy as np


class QuantumClassifier:
    def __init__(self, preprocessed_data, number_of_qubits):
        self.preprocessed_data = preprocessed_data
        self.number_of_qubits = number_of_qubits

    def get_angles_for_state_preparation(self):
        return np.array([QuantumClassifier.get_angles_from_data(data,self.number_of_qubits) for data in self.preprocessed_data])

    @staticmethod
    def get_angles_from_data(preprocessed_data,number_of_qubits):
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
            M = QuantumClassifier.get_multiplication_matrix(j+1, log2(j+1))
            rotation_angles.insert(0, np.matmul(M,alpha_jk))
        return rotation_angles

    @staticmethod
    def get_multiplication_matrix(size, number_of_controls):
        M = [[0 for i in range(size)] for j in range(size)]
        for i in range(size):
            for j in range(size):
                binary_j = bin(j).replace("0b", "")
                gray_i = bin(i ^ (i >> 1)).replace("0b", "")
                bitwise_product = 0
                for index in range(len(binary_j)):
                    bitwise_product += int(binary_j[index])*int(gray_i[index])
                M[i][j] = (-1)**bitwise_product
        M = 2**(-number_of_controls)*np.array(M)
        return M
