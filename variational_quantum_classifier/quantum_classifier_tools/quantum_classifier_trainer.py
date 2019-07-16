import funcsigs
import numpy as np

from scipy import optimize
from cirq import LineQubit, pauli_string_expectation, PauliString, Pauli, Simulator, measure

from .classifier_circuit_tools.state_preparation import StatePreparation
from .classifier_circuit_tools.model_circuit import ModelCircuit


class QuantumClassifierTrainer():
    def __init__(self, number_of_qubits, samples=None):
        self.number_of_qubits = number_of_qubits
        self.samples = samples
        self.qubits = [LineQubit(i) for i in range(self.number_of_qubits)]
        self.minimizer_kwargs = {'method': 'Nelder-Mead', 'options': {'disp': True, 'ftol': 1.0e-2, 'xtol': 1.0e-2}}

    def find_optimal_parameters(self, state_preparation_angles, original_labels, initial_classifier_parameters):

        def cost_function(classifier_parameters):
            predicted_labels = self.calculate_predictions(state_preparation_angles, classifier_parameters)
            return self.calculate_square_loss(original_labels, predicted_labels)

        args = [cost_function, initial_classifier_parameters]
        result = optimize.minimize(*args, **self.minimizer_kwargs)
        return result.x

    def calculate_quantum_classifier_accuracy(self, validation_angles, validation_labels, trained_classifier_parameters):
        validation_predictions = self.calculate_predictions(validation_angles, trained_classifier_parameters)
        validation_predictions = [np.sign(prediction.real) for prediction in validation_predictions]
        classifier_accuracy = self.calculate_accuracy(validation_labels, validation_predictions)
        return classifier_accuracy

    def calculate_predictions(self, state_preparation_angles, classifier_parameters):
        gate_parameters = classifier_parameters[0:6]
        bias = classifier_parameters[-1]
        pauli_z_expectations = [self.find_pauli_z_expectation(angle, gate_parameters)+bias for angle in state_preparation_angles]
        return pauli_z_expectations

    def find_pauli_z_expectation(self, angle, gate_parameters):
        variational_classifier_circuit = self.variational_quantum_classifier_circuit(angle, gate_parameters)        
        if self.samples is None:
            return self.derive_expectation_from_wavefunction(variational_classifier_circuit)
        else:
            #Todo
            variational_classifier_circuit.append([measure(self.qubits[0], key="q0")])
            return None

    def variational_quantum_classifier_circuit(self, angle, gate_parameters):
        state_preparation = StatePreparation(self.qubits)
        state_preparation_circuit = state_preparation.generate_state_preparation_circuit(angle)
        model_circuit = ModelCircuit(self.qubits)
        parameterized_model_circuit = model_circuit.get_parameterized_model_circuit()
        classifier_model_circuit = parameterized_model_circuit(gate_parameters)
        variational_classifier_circuit = state_preparation_circuit+classifier_model_circuit
        return variational_classifier_circuit

    def derive_expectation_from_wavefunction(self, variational_classifier_circuit):
        simulator = Simulator()
        simulation_result = simulator.simulate(variational_classifier_circuit)
        classifier_circuit_state = simulation_result.final_state
        pauli_z_operator = PauliString({self.qubits[0]: Pauli.by_index(2)}, 1)
        pauli_expectation = pauli_string_expectation(pauli_z_operator)
        return pauli_expectation.value_derived_from_wavefunction(classifier_circuit_state, {self.qubits[0]: 0})

    def calculate_square_loss(self, original_labels, predicted_labels):
        square_loss = 0
        for label, prediction in zip(original_labels, predicted_labels):
            square_loss = square_loss+(label-prediction)**2
        square_loss = square_loss/len(original_labels)
        return square_loss.real

    def calculate_accuracy(self, original_labels, predicted_labels):
        accuracy = 0
        for label, prediction in zip(original_labels, predicted_labels):
            if abs(label - prediction) < 1e-5:
                accuracy = accuracy + 1
        accuracy = accuracy / len(original_labels)
        return accuracy


