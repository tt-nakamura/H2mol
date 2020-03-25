import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from H2molUHF import H2molUHF

d = [1.39, 2.4]
x = np.linspace(-1.2,4,81)
y = np.linspace(-1.25,1.25,41)
x,y = np.meshgrid(x,y)

plt.figure(figsize=(6.4, 6.4))

for i,d1 in enumerate(d):
    r = minimize_scalar(lambda x: H2molUHF(d1,x).E, [1,1.4])
    h = H2molUHF(d1, r.x)
    plt.subplot(len(d),1,i+1)
    plt.contour(x,y,h.WaveFunc(x,y))
    plt.plot([0,d1],[0,0],'.k', markersize=8)
    plt.axis('equal')
    plt.axis([-1,4,-1.2,1.2])
    plt.yticks(np.arange(-1,2,step=1))
    plt.ylabel('y', fontsize=16)
    plt.text(3.3,1,'d = {:}'.format(d1), fontsize=14)
    plt.text(3.3,0.75,'Z = {:.2f}'.format(r.x), fontsize=14)

plt.xlabel('x', fontsize=16)
plt.tight_layout()
plt.savefig('fig5.eps')
plt.show()
