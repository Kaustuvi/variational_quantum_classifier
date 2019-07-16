from cirq import Circuit, SingleQubitMatrixGate, InsertStrategy, Rx, CZPowGate, CZ
import numpy as np
import math


class ModelCircuit:
    def __init__(self, qubits):
        self.number_of_qubits = len(qubits)
        self.range_of_control = find_range_controls(self.number_of_qubits)
        self.qubits = qubits
        self.parameterized_model_circuit = None

    def get_parameterized_model_circuit(self):
        def parameterized_model_circuit(parameters):
            parameterized_circuit = Circuit()
            for r in self.range_of_control:
                parameterized_circuit += self.get_single_qubit_gate_layer(
                    parameters[:3])
                parameterized_circuit += self.get_controlled_gate_layer(
                    parameters[3:], r)
            return parameterized_circuit
        return parameterized_model_circuit

    def get_single_qubit_gate_layer(self, params):
        single_qubit_gate_layer = Circuit()
        alpha = params[0]
        beta = params[1]
        gamma = params[2]
        single_qubit_gate = np.array([[np.exp(complex(0, beta))*np.cos(alpha), np.exp(complex(0, gamma))*np.sin(alpha)],
                                      [-1*np.exp(complex(0, -1*gamma))*np.sin(alpha), np.exp(complex(0, -1*beta))*np.cos(alpha)]])
        G = SingleQubitMatrixGate(single_qubit_gate)
        for qubit in self.qubits:
            single_qubit_gate_layer.append(
                [G(qubit)], strategy=InsertStrategy.EARLIEST)
        return single_qubit_gate_layer

    def get_controlled_gate_layer(self, params, r):
        controlled_gate_layer = Circuit()
        rot_param = params[0]
        controlled_phase_param = params[1]
        for k in range(self.number_of_qubits):
            RX = Rx(rot_param)
            CPhase = CZPowGate(exponent=controlled_phase_param/np.pi)
            controlled_gate_layer.append([RX(self.qubits[k])])
            controlled_gate_layer.append([CPhase(self.qubits[k*r % self.number_of_qubits], self.qubits[(
                k*r-r) % self.number_of_qubits])])
        return controlled_gate_layer


def find_range_controls(number_of_qubits):
    return [i for i in range(1, number_of_qubits) if i <= number_of_qubits/2 and math.gcd(i, number_of_qubits) == 1]
