from math import ceil, floor, log2
import numpy as np


class DataPreprocessor:
    def __init__(self, input_vectors):
        self.input_vectors = self.reshape_input_vectors(input_vectors)
        self.number_of_qubits = ceil(log2(len(self.input_vectors[0]))) if len(
            self.input_vectors.shape) == 2 else ceil(log2(len(self.input_vectors)))

    def reshape_input_vectors(self, input_vectors):
        if len(input_vectors.shape) == 2:
            return input_vectors
        return input_vectors.reshape(1, len(input_vectors))

    def pad_input_vectors(self):
        if is_equal_power_of_qubits(len(self.input_vectors[0]), self.number_of_qubits):
            return self.input_vectors

        padding_width = 2**self.number_of_qubits - len(self.input_vectors[0])
        padding_length = len(self.input_vectors)
        padding_constant = 0.3 * np.ones((padding_length, padding_width))
        return np.c_[self.input_vectors, padding_constant]

    def normalize_padded_vectors(self, padded_input_vectors):
        normalization = np.sqrt(np.sum(padded_input_vectors ** 2, -1))
        return (padded_input_vectors.T / normalization).T


def is_equal_power_of_qubits(dimension, number_of_qubits):
    return dimension == 2**number_of_qubits
