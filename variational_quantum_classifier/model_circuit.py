from cirq import Circuit, SingleQubitMatrixGate, InsertStrategy
import numpy as np
import math


class ModelCircuit:
    def __init__(self, state_preparation_circuit: Circuit):
        self.state_preparation_circuit = state_preparation_circuit
        self.number_of_qubits=len(self.state_preparation_circuit.all_qubits())
        self.qubits=self.state_preparation_circuit.all_qubits()
        self.parameterized_model_circuit = None

    def get_parameterized_model_circuit(self):
        def parameterized_model_circuit(parameters):
            range_of_control=find_range_contols(self.number_of_qubits)
            parameterized_circuit = Circuit()
            for r in range_of_control:
                parameterized_circuit += self.get_single_qubit_gate_layer(parameters[:3])
                # parameterized_circuit += self.get_controlled_gate_layer(parameters[3:])
            return parameterized_circuit
        return parameterized_model_circuit

    def get_single_qubit_gate_layer(self, params):
        single_qubit_gate_layer = Circuit()
        alpha=params[0]
        beta=params[1]
        gamma=params[2]
        single_qubit_gate=np.array([[np.exp(complex(0,beta))*np.cos(alpha),np.exp(complex(0,gamma))*np.sin(alpha)],
                            [-1*np.exp(complex(0,-1*gamma))*np.sin(alpha),-1*np.exp(complex(0,-1*beta))*np.cos(alpha)]])
        G=SingleQubitMatrixGate(single_qubit_gate)
        for qubit in self.qubits:
            single_qubit_gate_layer.append([G(qubit)],strategy=InsertStrategy.EARLIEST)
        return single_qubit_gate_layer

    def get_controlled_gate_layer(self, params):
        controlled_gate_layer = Circuit()
        return controlled_gate_layer

def find_range_contols(number_of_qubits):
    return [i for i in range(number_of_qubits) if i<=number_of_qubits/2 and math.gcd(i,number_of_qubits)==1]
