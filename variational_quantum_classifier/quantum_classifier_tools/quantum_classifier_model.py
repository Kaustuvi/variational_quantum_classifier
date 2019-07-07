from cirq import LineQubit

from .classifier_circuit_tools.state_preparation import StatePreparation
from .classifier_circuit_tools.model_circuit import ModelCircuit


class QuantumClassifierTrainer():
    def __init__(self, number_of_qubits):
        self.number_of_qubits = number_of_qubits
        self.qubits = [LineQubit(i) for i in range(self.number_of_qubits)]

    def find_optimal_parameters(self, state_preparation_angles, initial_classifier_parameters):
        variational_classifier_circuit=[self.variational_quantum_classifier_circuit(angle, initial_classifier_parameters[0]) for angle in state_preparation_angles]
        for circuit in variational_classifier_circuit:
            print(circuit)

    def variational_quantum_classifier_circuit(self, state_preparation_angles, gate_parameters):
        state_preparation = StatePreparation(self.qubits)
        state_preparation_circuit = state_preparation.generate_state_preparation_circuit(state_preparation_angles)
        model_circuit = ModelCircuit(self.qubits)
        parameterized_model_circuit = model_circuit.get_parameterized_model_circuit()
        classifier_model_circuit = parameterized_model_circuit(gate_parameters)
        variational_classifier_circuit = state_preparation_circuit+classifier_model_circuit
        return variational_classifier_circuit
