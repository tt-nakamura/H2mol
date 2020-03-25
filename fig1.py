import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from H2mol import H2mol

# distance at which energy takes minimum for theta>0
d = newton(lambda x: H2mol(x).eta()-1, 2)
print(d)

d = [1, 2, 3, 4]
theta = np.linspace(0, np.pi/4, 50)

for d1 in d:
    h = H2mol(d1)
    plt.plot(theta, h.energy(theta),
             label='d = {:}'.format(d1))

plt.legend(loc='upper left')
plt.axis([theta[0], theta[-1], -1.1,-0.5])
plt.xlabel(r'$\theta$  / radian', fontsize=16)
plt.ylabel(r'E = total energy of H$_2$  / Hartree', fontsize=16)
plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
