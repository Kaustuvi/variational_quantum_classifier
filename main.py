from cirq import Pauli, pauli_string_expectation, PauliString, LineQubit, Circuit, X, Simulator, measure,H
import numpy as np
from variational_quantum_classifier.variational_quantum_classifier import VariationalQuantumClassifier, split_data_for_classifier, get_classifier_data

raw_input_data = np.loadtxt("./data.txt")
original_features = raw_input_data[:,0:2]
original_labels = raw_input_data[:,-1]

vqc = VariationalQuantumClassifier(original_features)
state_preparation_angles = vqc.preprocess_raw_data()

training_data_indices, validation_data_indices = split_data_for_classifier(raw_input_data)
training_angles, training_labels = get_classifier_data(training_data_indices, state_preparation_angles, original_labels)
validation_angles, validation_labels = get_classifier_data(validation_data_indices, state_preparation_angles, original_labels)
classifier_parameters = vqc.train_quantum_classifier(training_angles, training_labels)

accuracy = vqc.validate_trained_classifier(validation_angles, validation_labels, classifier_parameters)
print(accuracy)

#reqd
# c=Circuit()
# c.append([H(LineQubit(0))])
# c.append([measure(LineQubit(0),key="q0")])
# print(c)
# sim=Simulator()
# result=sim.run(c, repetitions=10)
# # state=result.final_state
# print(result.measurements["q0"])
#reqd

# p=PauliString({LineQubit(0):Pauli.by_index(2)},1)
# pe=pauli_string_expectation(p, num_samples=None)
# print(pe.value_derived_from_wavefunction(state,{LineQubit(0):0}))
