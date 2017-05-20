import numpy as np
import matplotlib.pyplot as plt

h_bar = 6.626 * 10**(-34)/6.283
delta_t = 1.
tf = 3000.
gamma = 0.01
H = np.array([[0., 0.], [0., 0.]])
c = np.array([[1., 0.], [0., -1.]]) #Jumping operator. Also try [[0., 0.], [1., 0.]]
cT = np.transpose(c)
N = 0

trajectories = []
traj2 = []
r = 0
Nmax = 100 #number of trajectories to average over
while N < Nmax:
    psi = np.array([np.sqrt(2./3.), np.sqrt(1./3.)]) #Initial state. Also try [1., 0.]
    a = []
    b = []
    aa = []
    bb = []
    ab = []
    for i in range(int(tf / delta_t)):
        p = gamma * delta_t * np.linalg.norm(c.dot(psi))
        r = np.random.random(1)
        if r[0] > p:
            temp = np.copy(psi)
            psi = psi - (gamma / 2) * (cT.dot(c)).dot(psi) * delta_t  # 1j * H/h_bar = 0
            psi = psi / np.linalg.norm(psi)
        else:
            psi = c.dot(psi) / np.linalg.norm(c.dot(psi))
        a.append(psi[0])
        b.append(psi[1])
        aa.append(psi[0]**2)
        bb.append(psi[1]**2)
        ab.append(psi[0]*psi[1])
    trajectories.append(a)
    traj2.append(b)

    N = N + 1

t = []
avg_a = [sum(x) for x in zip(*trajectories)]
avg_b = [sum(x) for x in zip(*traj2)]

for i in range(int(tf / delta_t)):
    t.append((i + 1) * delta_t * gamma)
    avg_a[i] = avg_a[i] / Nmax
    avg_b[i] = avg_b[i] / Nmax

plt.plot(t, avg_a, 'r', t, avg_b, 'b')
plt.show()


