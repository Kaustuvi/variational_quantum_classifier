import time
import numpy as np
from variational_quantum_classifier.variational_quantum_classifier import VariationalQuantumClassifier

input_data = np.loadtxt("./data/iris_data.txt")
raw_features = input_data[:, 0:2]
original_labels = input_data[:, -1]

vqc = VariationalQuantumClassifier(raw_features, samples=10000)
start_time = time.time()
state_preparation_angles = vqc.preprocess_raw_data()

training_data_indices, validation_data_indices = vqc.split_data_for_classifier(input_data)
training_angles, training_labels = vqc.get_classifier_data(training_data_indices, state_preparation_angles, original_labels)
validation_angles, validation_labels = vqc.get_classifier_data(validation_data_indices, state_preparation_angles, original_labels)

print("\n\tTraining Results of Variational Quantum Classifier (per iteration) -->\n")
optimal_classifier_parameters, overall_cost = vqc.train_quantum_classifier(training_angles, training_labels)
end_time = time.time()
training_time = end_time - start_time

accuracy = vqc.validate_trained_classifier(validation_angles, validation_labels, optimal_classifier_parameters)
print("\n\tOverall accuracy of Quantum Classifier for Validation Data: ", accuracy)
print("\tOverall cost of Quantum Classifier: {:6.3f}".format(overall_cost))
print("\tTraining Time of Quantum Classifier: {:6.3f} seconds".format(training_time))

new_raw_features = vqc.get_new_raw_features()
vqc = VariationalQuantumClassifier(new_raw_features)
new_state_preparation_angles = vqc.preprocess_raw_data()
predicted_labels = vqc.predict_new_labels(new_state_preparation_angles, optimal_classifier_parameters)
vqc.plot_decision_regions(input_data, original_labels, predicted_labels)
