from qclassify.preprocessing import *
from qclassify.encoding_circ import *

from pyquil.gates import RY, CNOT, X
from pyquil import Program
from pyquil.api import WavefunctionSimulator
import numpy as np
import pennylane as pl

x = np.array([0.5, np.sqrt(3/8), 0.5, np.sqrt(1/8)])
beta0 = 2 * np.arcsin(np.sqrt(x[1]) ** 2 /
                      np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
beta1 = 2 * np.arcsin(np.sqrt(x[3]) ** 2 /
                      np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
beta2 = 2 * np.arcsin(np.sqrt(x[2] ** 2 + x[3] ** 2) /
                      np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2))
a = np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])
print(x)
print(a)
p = Program()
p += RY(beta2, 0)
p += RY((beta0+beta1)/2, 1)
# p += CNOT(0, 1)
# p += RY((beta0-beta1)/2, 1)
# p += CNOT(0, 1)
wf = WavefunctionSimulator()
print(wf.wavefunction(p))
pl.expval.PauliZ(0)
