from cirq import Circuit, GridQubit, Ry, CNOT, InsertStrategy, Simulator
import numpy as np
from variational_quantum_classifier.data_preprocessor import DataPreprocessor
from variational_quantum_classifier.state_preparation import StatePreparation

size = 2
qubits = [GridQubit(i, j) for i in range(size) for j in range(size)]

data = np.loadtxt("./data.txt")
X = data
print(X)
# X = X.reshape(1, 4)
# x = X[0]
# beta0 = 2 * np.arcsin(np.sqrt(x[1]) ** 2 /
#                       np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
# beta1 = 2 * np.arcsin(np.sqrt(x[3]) ** 2 /
#                       np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
# beta2 = 2 * np.arcsin(np.sqrt(x[2] ** 2 + x[3] ** 2) /
#                       np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2))
# a = np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])
# print(beta2,beta0,beta1)
# print(beta2,(beta0+beta1)/2,(beta0-beta1)/2)
# ckt = Circuit()
# RY = Ry(beta2)
# ckt.append([RY(GridQubit(0, 0))], strategy=InsertStrategy.EARLIEST)
# RY = Ry((beta0+beta1)/2)
# ckt.append([RY(GridQubit(0, 1))], strategy=InsertStrategy.EARLIEST)
# ckt.append([CNOT(GridQubit(0, 0),GridQubit(0,1))], strategy=InsertStrategy.EARLIEST)
# RY = Ry((beta0-beta1)/2)
# ckt.append([RY(GridQubit(0, 1))], strategy=InsertStrategy.EARLIEST)
# ckt.append([CNOT(GridQubit(0, 0),GridQubit(0,1))], strategy=InsertStrategy.EARLIEST)
# sim = Simulator()
# result = sim.simulate(ckt)
# print(result.final_state)

data_preprocessor = DataPreprocessor(X)
padded_input_vectors = data_preprocessor.pad_input_vectors()
normalized_padded_vectors = data_preprocessor.normalize_padded_vectors(padded_input_vectors)
quantum_classifier = StatePreparation(normalized_padded_vectors, data_preprocessor.number_of_qubits)
angles = quantum_classifier.get_angles_for_state_preparation()
quantum_classifier.generate_state_preparation_circuit()
print(quantum_classifier.state_preparation_circuit)

# binary and gray bitwise dot product code
# b = 5
# g = b ^ (b >> 1)
# b = bin(b).replace("0b", "")
# g = bin(g).replace("0b", "")
# p=0
# for i in range(len(b)):
#     p += int(b[i])*int(g[i])
# print(b)
# print(g)
# print(p)
