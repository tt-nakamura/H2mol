import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from H2molUHF import H2molUHF

d = np.geomspace(0.6, 5, 50)
E = []

for d1 in d:
    r = minimize_scalar(lambda x: H2molUHF(d1,x).E, [1,1.5])
    E.append(r.fun)

plt.axis([0,d[-1],-1.13,-0.9])
plt.plot(d,E)
plt.xlabel('d = distance between protons  / Bohr radius', fontsize=16)
plt.ylabel(r'E = total energy of H$_2$  / Hartree', fontsize=16)
plt.tight_layout()
plt.savefig('fig4.eps')
plt.show()
