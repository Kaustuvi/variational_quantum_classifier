from math import ceil, log2
import numpy as np


class DataPreprocessor:
    def __init__(self, raw_input_data):
        self.raw_input_data = raw_input_data
        self.number_of_qubits = None
        self.state_preparation_angles = None

    def pad_input_vectors(self, reshaped_input_vectors):
        if is_equal_power_of_qubits(len(reshaped_input_vectors[0]), self.number_of_qubits):
            return reshaped_input_vectors
        padding_width = 2**self.number_of_qubits - len(reshaped_input_vectors[0])
        padding_length = len(reshaped_input_vectors)
        padding_constant = 0.3 * np.ones((padding_length, padding_width))
        return np.c_[reshaped_input_vectors, padding_constant]

    def normalize_padded_vectors(self, padded_input_vectors):
        normalization = np.sqrt(np.sum(padded_input_vectors ** 2, -1))
        return (padded_input_vectors.T / normalization).T

    def reshape_input_vectors(self):
        if len(self.raw_input_data.shape) == 2:
            return self.raw_input_data
        return self.raw_input_data.reshape(1, len(self.raw_input_data))

    def get_number_of_qubits(self, reshaped_input_vectors):
        return ceil(log2(len(reshaped_input_vectors[0]))) if len(reshaped_input_vectors.shape) == 2 else ceil(log2(len(reshaped_input_vectors)))

    def preprocess_input_data(self):
        reshaped_input_vectors = self.reshape_input_vectors()
        self.number_of_qubits = self.get_number_of_qubits(reshaped_input_vectors)
        padded_data = self.pad_input_vectors(reshaped_input_vectors)
        preprocessed_input_data = self.normalize_padded_vectors(padded_data)
        return preprocessed_input_data

    def get_angles_for_state_preparation(self, preprocessed_input_data):
        self.state_preparation_angles = [self.get_angles_from_data(data, self.number_of_qubits) for data in preprocessed_input_data]
        return self.state_preparation_angles, self.number_of_qubits

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


def is_equal_power_of_qubits(dimension, number_of_qubits):
    return dimension == 2**number_of_qubits


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
