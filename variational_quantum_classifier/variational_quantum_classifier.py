import numpy as np

from .quantum_classifier_tools.data_preprocessor import DataPreprocessor
from .quantum_classifier_tools.quantum_classifier_model import QuantumClassifierTrainer


class VariationalQuantumClassifier():
    def __init__(self, raw_input_data):
        self.raw_input_data = raw_input_data
        self.number_of_qubits = None

    def train_quantum_classifier(self):
        # preprocess raw input data
        state_preparation_angles, self.number_of_qubits = self.preprocess_raw_data(self.raw_input_data)

        # find optimal parameters for variational quantum classifier
        classifier_parameters = self.find_classifier_parameters(state_preparation_angles, self.number_of_qubits)
        return classifier_parameters

    @staticmethod
    def preprocess_raw_data(raw_input_data):
        data_preprocessor = DataPreprocessor(raw_input_data)
        preprocessed_input_data = data_preprocessor.preprocess_input_data()
        state_preparation_angles, number_of_qubits = data_preprocessor.get_angles_for_state_preparation(preprocessed_input_data)
        return state_preparation_angles, number_of_qubits

    @staticmethod
    def find_classifier_parameters(state_preparation_angles, number_of_qubits):
        gate_parameters = [np.random.uniform(0, np.pi)]*5
        bias = np.random.uniform(0, np.pi)
        initial_classifier_parameters = [gate_parameters, bias]
        qcm_inst = QuantumClassifierTrainer(number_of_qubits)
        return qcm_inst.find_optimal_parameters(state_preparation_angles, initial_classifier_parameters)
