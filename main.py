from cirq import Circuit, GridQubit, Ry, CNOT, InsertStrategy, Simulator, CZ, LineQubit, Rx
import numpy as np
from variational_quantum_classifier.variational_quantum_classifier import VariationalQuantumClassifier

raw_input_data = np.loadtxt("./data.txt")
vqc = VariationalQuantumClassifier(raw_input_data)
classifier_parameters = vqc.train_quantum_classifier()
#preprocess input data
# reshaped_input_vectors=reshape_input_vectors(X)
# number_of_qubits=get_number_of_qubits(reshaped_input_vectors)
# data_preprocessor = DataPreprocessor(reshaped_input_vectors, number_of_qubits)
# preprocessed_input_data=data_preprocessor.preprocess_input_data()
# state_prep_angles = data_preprocessor.get_angles_for_state_preparation()
# vqc=VariationalQuantumClassifier(number_of_qubits)
# variational_classfier_circuit=vqc.variational_classifier(state_prep_angles)
# model_circuit = ModelCircuit(state_preparation.qubits)
# parameterized_model_circuit=model_circuit.get_parameterized_model_circuit()
# ckt=parameterized_model_circuit([np.pi/2,np.pi/2,np.pi/2,1,2])
# sim=Simulator()
# print(ckt)
# res=sim.simulate(ckt)
# print(res.dirac_notation())
