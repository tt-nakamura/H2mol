import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from H2mol import H2mol

r = minimize_scalar(lambda x: H2mol(x).E_min(), [1,2])
print('proton separation: {:6g}'.format(r.x))
print('minimum energy: {:6g}'.format(r.fun))

d = np.geomspace(0.8, 5, 50)
E = []

for d1 in d:
    E.append(H2mol(d1).E_min())

plt.plot(d,E)
plt.axis([0, d[-1], -1.13, -0.9])
plt.xlabel('d = distance between protrons  / Bohr radius', fontsize=16)
plt.ylabel(r'E = total energy of H$_2$  / Hartree', fontsize=16)
plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()
