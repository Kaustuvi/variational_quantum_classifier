import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from .quantum_classifier_tools.data_preprocessor import DataPreprocessor
from .quantum_classifier_tools.quantum_classifier_trainer import QuantumClassifierTrainer


class VariationalQuantumClassifier():
    def __init__(self, raw_features, samples=None):
        self.raw_features = raw_features
        self.number_of_qubits = None
        self.samples = samples
        self.qcm_inst = None

    def preprocess_raw_data(self):
        data_preprocessor = DataPreprocessor(self.raw_features)
        preprocessed_input_data = data_preprocessor.preprocess_input_data()
        state_preparation_angles, self.number_of_qubits = data_preprocessor.get_angles_for_state_preparation(preprocessed_input_data)
        self.qcm_inst = QuantumClassifierTrainer(self.number_of_qubits, self.samples)
        return state_preparation_angles

    def train_quantum_classifier(self, training_angles, training_labels):
        initial_classifier_parameters = [np.random.uniform(0, np.pi)]*6
        classifier_parameters = self.qcm_inst.find_optimal_parameters(training_angles, training_labels, initial_classifier_parameters)
        return classifier_parameters

    def validate_trained_classifier(self, validation_angles, validation_labels, trained_classifier_parameters):
        classifier_accuracy = self.qcm_inst.calculate_quantum_classifier_accuracy(validation_angles, validation_labels, trained_classifier_parameters)
        return classifier_accuracy

    def find_decision_regions(self):
        pass

def split_data_for_classifier(raw_input_data):
    total_number_of_data = len(raw_input_data)
    number_of_training_data = int(0.75*total_number_of_data)
    indices = np.random.permutation(range(total_number_of_data))
    return indices[:number_of_training_data], indices[number_of_training_data:]

def get_classifier_data(random_indices, state_preparation_angles, original_labels):
    training_angles = []
    training_labels = []
    for index in random_indices:
        training_angles.append(state_preparation_angles[index])
        training_labels.append(original_labels[index])
    return training_angles, training_labels

