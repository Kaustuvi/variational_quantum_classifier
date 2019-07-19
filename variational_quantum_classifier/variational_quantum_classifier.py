import numpy as np
import matplotlib.pyplot as plt

from .quantum_classifier_tools.data_preprocessor import DataPreprocessor
from .quantum_classifier_tools.quantum_classifier_trainer import QuantumClassifierTrainer


class VariationalQuantumClassifier():
    def __init__(self, raw_features, samples=None):
        self.raw_features = raw_features
        self.x_axis, self.y_axis = np.meshgrid(np.linspace(0.01, 1.5, 20), np.linspace(0.01, 1.5, 20))
        self.number_of_qubits = None
        self.samples = samples
        self.qct_inst = None

    def preprocess_raw_data(self):
        data_preprocessor = DataPreprocessor(self.raw_features)
        preprocessed_input_data = data_preprocessor.preprocess_input_data()
        state_preparation_angles, self.number_of_qubits = data_preprocessor.get_angles_for_state_preparation(preprocessed_input_data)
        self.qct_inst = QuantumClassifierTrainer(self.number_of_qubits, self.samples)
        return state_preparation_angles

    def split_data_for_classifier(self, input_data):
        total_number_of_data = len(input_data)
        number_of_training_data = int(0.75*total_number_of_data)
        indices = np.random.permutation(range(total_number_of_data))
        return indices[:number_of_training_data], indices[number_of_training_data:]

    def get_classifier_data(self, random_indices, state_preparation_angles, original_labels):
        angles = []
        labels = []
        for index in random_indices:
            angles.append(state_preparation_angles[index])
            labels.append(original_labels[index])
        return angles, labels

    def train_quantum_classifier(self, training_angles, training_labels):
        initial_classifier_parameters = [np.random.uniform(0, np.pi)]*6
        classifier_parameters, overall_cost = self.qct_inst.find_optimal_parameters(training_angles, training_labels, initial_classifier_parameters)
        return classifier_parameters, overall_cost

    def validate_trained_classifier(self, validation_angles, validation_labels, trained_classifier_parameters):
        classifier_accuracy = self.qct_inst.calculate_validation_label_accuracy(validation_angles, validation_labels, trained_classifier_parameters)
        return classifier_accuracy

    def get_new_raw_features(self):
        new_raw_features = np.array([np.array([x, y]) for x, y in zip(self.x_axis.flatten(), self.y_axis.flatten())])
        return new_raw_features

    def predict_new_labels(self, angles, parameters):
        return np.reshape(self.qct_inst.calculate_predictions(angles, parameters), self.x_axis.shape)

    def plot_decision_regions(self, input_data, original_labels, predicted_labels):
        plt.figure()
        plt.title("Plot of Training and Validation Data on Decision Regions", pad=15.5)
        cm = plt.cm.get_cmap('RdBu')
        cnt = plt.contourf(self.x_axis, self.y_axis, predicted_labels, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=.8, extend='both')
        plt.contour(self.x_axis, self.y_axis, predicted_labels, levels=[0.0], colors=('black',), linestyles=('--',), linewidths=(0.8,))
        plt.colorbar(cnt, ticks=[-1, 0, 1])
        training_data_indices, validation_data_indices = self.split_data_for_classifier(input_data)
        X_train, Y_train = self.get_classifier_data(training_data_indices, input_data[:,0:2], original_labels)
        X_val, Y_val = self.get_classifier_data(validation_data_indices, input_data[:,0:2], original_labels)
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_val, Y_val = np.array(X_val), np.array(Y_val)
        plt.scatter(X_train[:,0][Y_train==1], X_train[:,1][Y_train==1], c='b', marker='o', edgecolors='k', label="class 1 train")
        plt.scatter(X_val[:,0][Y_val==1], X_val[:,1][Y_val==1], c='b', marker='^', edgecolors='k', label="class 1 validation")
        plt.scatter(X_train[:,0][Y_train==-1], X_train[:,1][Y_train==-1], c='r', marker='o', edgecolors='k', label="class -1 train")
        plt.scatter(X_val[:,0][Y_val==-1], X_val[:,1][Y_val==-1], c='r', marker='^', edgecolors='k', label="class -1 validation")
        plt.legend()
        plt.show()

