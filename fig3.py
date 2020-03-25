import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from H2molUHF import H2molUHF

r = minimize(lambda x: H2molUHF(*x).E, [1.4, 1.3])
d0,Z0 = r.x

print('nuclear separation: {:6g}'.format(d0))
print('effective charge: {:6g}'.format(Z0))
print('binding energy: {:6g}'.format(r.fun))

d = np.linspace(1.2,1.6,50)
Z = np.linspace(1,1.6,50)

E = []
for d1 in d:
    E1 = []
    for Z1 in Z:
        E1.append(H2molUHF(d1,Z1).E)
    E.append(E1)

x,y = np.meshgrid(d,Z,indexing='ij')
levels = np.linspace(np.min(E),np.max(E),21)

plt.contour(x,y,E,levels=levels)
plt.plot([d0],[Z0],'+k',markersize=8)
plt.axis([d[0],d[-1],Z[0],Z[-1]])
plt.xlabel('d = distance between protons  / Bohr radius', fontsize=16)
plt.ylabel(r'Z = effective charge of a proton', fontsize=16)
plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()
