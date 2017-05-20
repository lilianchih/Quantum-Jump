import numpy as np
import matplotlib.pyplot as plt

delta_t = 1.
tf = 3000.
gamma = 0.01
xi = 0.5 * gamma
V = np.array([[0., xi, 0.], [xi,  0., xi], [0., xi, 0.]]) #Suppose detuning = 0
c = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
cT = np.transpose(c)
N = 0

traj_ee = []
traj_mg = []
traj_mm = []
r = 0
Nmax = 100
while N < Nmax:
    psi = np.array([1., 0., 0.], dtype = complex)
    ee = []
    mg = []
    mm = []
    for i in range(int(tf / delta_t)):
        p = gamma * delta_t * np.linalg.norm(c.dot(psi))
        r = np.random.random(1)
        if r[0] > p:
            psi = psi - (1j * V.dot(psi) - (gamma / 2) * (cT.dot(c)).dot(psi)) * delta_t
            psi = psi / np.linalg.norm(psi)
        else:
            psi = c.dot(psi) / np.linalg.norm(c.dot(psi))
        ee.append(np.linalg.norm(psi[0])**2)
        mg.append(psi[1]*np.conjugate(psi[2]))
        mm.append(np.linalg.norm(psi[1])**2)
    traj_ee.append(ee)
    traj_mg.append(mg)
    traj_mm.append(mm)
    N = N + 1

t = []
avg_ee = [sum(x) for x in zip(*traj_ee)]
avg_mg = [sum(x) for x in zip(*traj_mg)]
avg_mm = [sum(x) for x in zip(*traj_mm)]

for i in range(int(tf / delta_t)):
    t.append((i + 1) * delta_t * gamma)
    avg_ee[i] = avg_ee[i] / Nmax
    avg_mg[i] = avg_mg[i] * (-1j) / Nmax #get (purely) imaginary part
    avg_mm[i] = avg_mm[i] / Nmax

plt.plot(t, avg_ee, 'r', t, avg_mm, 'b', t, avg_mg, 'c')
plt.show()