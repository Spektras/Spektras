import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
s = np.loadtxt('Horizontal_flow.txt')
n = np.loadtxt('electron_density.txt')
N = np.loadtxt('Emission_electron_density.txt')
ds = np.insert(np.diff(1000.*s),0,0)
dn = np.insert(np.diff(n),0,0)
dN = np.insert(np.diff(N),0,0)
n_H = np.nan_to_num((10**12*dn)**(1/3))
N_H = (10**12*(-dN))**(1/3)
r_min = ds/n_H
r_min[np.isinf(r_min)]=0
r_min = np.nan_to_num(r_min)
r_max = (ds/N_H)**-1
r_f = 1/len(s)*(np.nan_to_num(2/3*r_min**2+1/3*r_max**2))
R_max = (1/len(s)*(np.nan_to_num(2/3*r_min**2+1/3*r_max**2))**(1/2))/10**2
r_H = (len(s)**2*ds**2*n_H+10**4*len(s)*ds**2-len(s)*n_H**2-2*10**8*n_H)/(len(s)**2*(ds*n_H**2-ds*n_H*N_H))
r_H[np.isinf(r_H)]=0
R_H = np.abs(np.gradient(1000*s+np.nan_to_num(r_H))/1000)
np.savetxt('r_f.txt',r_f)
R_min = r_min*10**2
N_yield = N_H/N
geometry = [R_max,R_min,R_H,N]
np.savetxt('geometry.txt',geometry)
N0 = max(N_H)
R_max0 = R_max[np.asarray(np.where(N_H==max(N_H)),dtype='int')]
R_min0 = R_min[np.asarray(np.where(R_min==min(R_min)),dtype='int')]
R_H0 = R_H[np.asarray(np.where(N_H<10**2),dtype='int')]
geometry_meteoroid = [R_max0,R_min0,R_H0,N0]
np.savetxt('geometry_meteoroid.txt',geometry_meteoroid)
f = interpolate.interp2d(R_max,N,R_min)
g = interpolate.interp2d(R_min,N,R_max)
h = interpolate.interp2d(R_min,R_max,N)
i = interpolate.interp2d(R_max,R_H,R_min)
j = interpolate.interp2d(R_min,R_H,R_max)
k = interpolate.interp2d(R_min,R_max,R_H)
F = f(R_max,N)
G = g(R_min,N)
H = h(R_min,R_max)
I = i(R_max,R_H)
J = j(R_min,R_H)
K = k(R_min,R_max)
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(F,G,H,c='r'),ax.plot_wireframe(F,G,H)
ax.set_xlabel('a (m)')
ax.set_ylabel('b (m)')
ax.set_zlabel('n (1/cm3)')
fig1.savefig('Plasma_geometry(x).png')
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(G,F,H,c='r'),ax.plot_wireframe(G,F,H)
ax.set_xlabel('b (m)')
ax.set_ylabel('a (m)')
ax.set_zlabel('n (1/cm3)')
fig2.savefig('Plasma_geometry(y).png')
fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
ax.scatter(I,J,K,c='r'),ax.plot_wireframe(I,J,K)
ax.set_xlabel('a (m)')
ax.set_ylabel('b (m)')
ax.set_zlabel('H (m)')
fig3.savefig('Plasma_geometry.png')
fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')
ax.scatter(R_min0,R_max0,R_H0, c='r')
ax.set_xlabel('a (m)')
ax.set_ylabel('b (m)')
ax.set_zlabel('H (m)')
fig4.savefig('Meteoroid_geometry.png')
