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
        self.minimizer_kwargs = {'method': 'Nelder-Mead', 'options': {'disp': False, 'ftol': 1.0e-2, 'xtol': 1.0e-2}}
        self.current_cost = None
        self.current_accuracy = None

    def find_optimal_parameters(self, state_preparation_angles, original_labels, initial_classifier_parameters):

        def cost_function(classifier_parameters):
            predicted_labels = self.calculate_predictions(state_preparation_angles, classifier_parameters)
            current_predictions = [np.sign(prediction) for prediction in predicted_labels]
            self.current_cost = self.calculate_square_loss(original_labels, predicted_labels)
            self.current_accuracy = self.calculate_accuracy(original_labels, current_predictions)
            return self.calculate_square_loss(original_labels, predicted_labels)

        def print_current_iteration(iteration_variables):
            print("\tCost: {:6.3f} \tOverall Accuracy: {:6.3f}".format(self.current_cost, self.current_accuracy))
        
        self.minimizer_kwargs['callback'] = print_current_iteration
        args = [cost_function, initial_classifier_parameters]
        result = optimize.minimize(*args, **self.minimizer_kwargs)
        return result.x, result.fun

    def calculate_validation_label_accuracy(self, validation_angles, validation_labels, trained_classifier_parameters):
        validation_predictions = self.calculate_predictions(validation_angles, trained_classifier_parameters)
        validation_predictions = [np.sign(prediction) for prediction in validation_predictions]
        self.print_accuracy_table(validation_labels, validation_predictions)
        classifier_accuracy = self.calculate_accuracy(validation_labels, validation_predictions)
        return classifier_accuracy

    def calculate_predictions(self, state_preparation_angles, classifier_parameters):
        gate_parameters = classifier_parameters[0:6]
        bias = classifier_parameters[-1]
        pauli_z_expectations = [self.find_pauli_z_expectation(angle, gate_parameters).real+bias for angle in state_preparation_angles]
        return pauli_z_expectations

    def find_pauli_z_expectation(self, angle, gate_parameters):
        variational_classifier_circuit = self.variational_quantum_classifier_circuit(angle, gate_parameters)
        if self.samples is None:
            return self.derive_expectation_from_wavefunction(variational_classifier_circuit)
        else:
            variational_classifier_circuit.append([measure(self.qubits[0], key="q0")])
            return self.derive_expectation_from_samples(variational_classifier_circuit)

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

    def derive_expectation_from_samples(self, variational_classifier_circuit):
        simulator = Simulator()
        simulation_result = simulator.run(variational_classifier_circuit, repetitions=self.samples)
        classifier_circuit_state = simulation_result.measurements["q0"]
        pauli_z_operator = PauliString({self.qubits[0]: Pauli.by_index(2)}, 1)
        pauli_expectation = pauli_string_expectation(pauli_z_operator, num_samples=self.samples)
        return pauli_expectation.value_derived_from_samples(classifier_circuit_state)

    def calculate_square_loss(self, original_labels, predicted_labels):
        square_loss = 0
        for label, prediction in zip(original_labels, predicted_labels):
            square_loss = square_loss+(label-prediction)**2
        square_loss = square_loss/len(original_labels)
        return square_loss

    def calculate_accuracy(self, original_labels, predicted_labels):
        accuracy = 0
        for label, prediction in zip(original_labels, predicted_labels):
            if abs(label - prediction) < 1e-5:
                accuracy = accuracy + 1
        accuracy = accuracy / len(original_labels)
        return accuracy

    def print_accuracy_table(self, original_labels, predicted_labels):
        print("\n\tAccuracy of each predicted label for validation data -->\n")
        print("\tOriginal Label\t\tPredicted Label\t\tAccuracy")
        for label, prediction in zip(original_labels, predicted_labels):
            print("\t{:4.1f}\t{:20.1f}\t{:20.1f}".format(label, prediction, abs(label/prediction)))

