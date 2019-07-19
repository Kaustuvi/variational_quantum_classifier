from cirq import Circuit, InsertStrategy, Ry, CNOT
from sympy.combinatorics.graycode import GrayCode


class StatePreparation:
    def __init__(self, qubits):
        self.qubits = qubits
        self.state_preparation_circuit = None

    def generate_state_preparation_circuit(self, state_preparation_angles):
        state_preparation_circuit = Circuit()
        for angles in state_preparation_angles:
            qubit_index = state_preparation_angles.index(angles)
            if(qubit_index == 0):
                qubit_angle = angles[qubit_index]
                RY = Ry(qubit_angle)
                state_preparation_circuit.append([RY(self.qubits[qubit_index])], strategy=InsertStrategy.EARLIEST)
            else:
                gray_code_list = generate_gray_code(qubit_index)
                for qubit_angle in angles:
                    RY = Ry(qubit_angle)
                    state_preparation_circuit.append([RY(self.qubits[qubit_index])], strategy=InsertStrategy.EARLIEST)
                    l = angles.index(qubit_angle)
                    cnot_position = self.find_cnot_position(gray_code_list[(l+1) % len(angles)], gray_code_list[l % len(angles)])
                    state_preparation_circuit += self.get_cnot_circuit(self.qubits[cnot_position[0]], self.qubits[qubit_index])
        self.state_preparation_circuit = state_preparation_circuit
        return self.state_preparation_circuit

    @staticmethod
    def get_cnot_circuit(control_line, target_line):
        cnot_circuit = Circuit()
        cnot_circuit.append([CNOT(control_line, target_line)], strategy=InsertStrategy.EARLIEST)
        return cnot_circuit

    @staticmethod
    def find_cnot_position(curr_gray_code, prev_gray_code):
        return [i for i in range(len(curr_gray_code)) if curr_gray_code[i] != prev_gray_code[i]]


def generate_gray_code(number_of_controls):
    return list(GrayCode(number_of_controls).generate_gray())
