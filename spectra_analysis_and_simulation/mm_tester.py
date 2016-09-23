import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.misc
import math
from scipy.misc import factorial
from mpl_toolkits.mplot3d import Axes3D
ablation = np.loadtxt('ablace.spa')
lambda_m_nm = ablation[:,0]
lambda_m = lambda_m_nm/10**9
I0 = ablation[:,1]
I = I0 + np.abs(min(I0))
k = np.arange(0,len(I),1,dtype='float')
k1 = np.loadtxt('input_one.txt')
T_min = 190.05
h = 6.626e-034
c = 3e008
k_B = 1.38e-023
lambda_De = 3.104e-005
N_D = 22.41
epsilon = 8.854e-012
e = 1.602e-19
wien = 2.898e-003
m_p = 1.6726e-027
m_e = 9.1094e-031
def mean(x):
	return np.sum(x)/len(x)
T0 = (wien/lambda_m-mean(wien/lambda_m)+np.abs(min(wien/lambda_m-mean(wien/lambda_m)))) - (np.abs(wien/lambda_m-mean(wien/lambda_m))+(max(np.abs(wien/lambda_m-mean(wien/lambda_m)))-np.abs(wien/lambda_m-mean(wien/lambda_m))))+np.abs(min((wien/lambda_m-mean(wien/lambda_m)+np.abs(min(wien/lambda_m-mean(wien/lambda_m)))) - (np.abs(wien/lambda_m-mean(wien/lambda_m))+(max(np.abs(wien/lambda_m-mean(wien/lambda_m)))-np.abs(wien/lambda_m-mean(wien/lambda_m))))))
E = h*c/lambda_m
a = k1**(k1-1)/factorial(k1)*(1/((T_min*(E*k_B)**(T_min))-np.log(T_min)))**k1
T0_1 = np.abs(np.log(1/(a+np.e)))
b = T_min/(np.log(T_min-I*E*k_B))
T1_1 = np.abs(T0_1*b)*scipy.misc.logsumexp(I)
T = (np.abs(np.gradient(np.gradient(T1_1)))+T0)**2/(2*T0)
dT = np.gradient(T)
np.savetxt('temperature.txt',T)
fig0 = plt.figure()
plt.plot(lambda_m_nm,T)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Temperature (K)')
plt.xlim(xmin=190)
plt.xlim(xmax=900)
fig0.savefig('Temperature_fit0.png')
fig1 = plt.figure()
plt.plot(T,I)
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Temperature (K)')
fig1.savefig('Temperature_fit.png')
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(lambda_m_nm,T,I, c='r')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Temperature (K)')
ax.set_zlabel('Intensity (a.u.)')
fig2.savefig('Spectra_temperature.png')
a = k1**(k1-1)/factorial(k1)*1/3*(9/16*(epsilon*k_B*np.e)*np.pi**-2*np.exp(-dT/(4*np.pi)))**k1+1
b = a*16/9*epsilon**3*k_B**2*np.pi**2*lambda_De**(6/5)*np.exp(dT/2.241)/(np.e**6*22.41**2)
n = (np.exp(-np.log(b*10**12)/20)*np.exp((b*10**12)/20))+np.exp((b*10**12)/20)
dN = np.abs(np.gradient(n))
np.savetxt('electron_density.txt',[n,dN])
B = 1+0.1*np.log(I/10**-12)
m_D = 1.76e-010
h_0 = 10*(np.pi*B*1/(2+2*m_D))
h_t = np.abs(B*scipy.misc.logsumexp(1/2*np.exp((b*10**12)/20)*B/(np.exp(n*(1-m_D))-(B*np.exp((b*10**12)/20)/(n*(1-m_D))**4)-4)**(1/2)))
s = (2*h_0**2)
ds = np.gradient(s)
dn = 1.72e+005
N = [n,n]
DN = [(n+dN),(n-dN)]
Te = [T,T]
fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
ax.scatter(Te,DN,N, c='r')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('dn/dT (1/cm3)')
ax.set_zlabel('Electron density (1/cm3)')
fig3.savefig('Plasma_distribution.png')
n_H = (10**12*dn)**(1/3)
N_H = (10**12*(-dN))**(1/3)
r_min = ds/n_H*1000
r_max = 1/2*np.abs((1/len(s)*(2/3*r_min**2+1/3*(1000/(ds/n_H))/s**2))**(1/2))*1000
r_f = 1/len(s)*(2/3*r_min**2+1/3*r_max**2)
np.savetxt('a.txt',r_min)
np.savetxt('b.txt',r_max)
np.savetxt('r_f.txt',r_f)
R_MAX = [r_max,r_max]
R_MIN = [r_min, -r_min]
N1 = [n/10**5, n/10**5]
fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')
ax.scatter(R_MIN,R_MAX,N1, c='r')
ax.set_xlabel('a (m)')
ax.set_ylabel('b (m)')
ax.set_zlabel('Electron density yield (n/n0)')
fig4.savefig('Plasma_geometry(x).png')
fig5 = plt.figure()
ax = fig5.add_subplot(111, projection='3d')
ax.scatter(R_MAX,R_MIN,N1, c='r')
ax.set_xlabel('a (m)')
ax.set_ylabel('b (m)')
ax.set_zlabel('Electron density yield (n/n0)')
fig5.savefig('Plasma_geometry(y).png')
geometry = [r_min, r_max, dN]
master1 = [lambda_m_nm, I, T]
np.savetxt('master1.txt',master1)
v_e = 4.19*10**7*T**1/2 #cm/s
v_Fe = 9.79*10**5*(9.2703e-026/m_p)**1/2*T**1/2 # mean q velocity of iron ion
s1_0 = 2*10**-7/7100/gamma(2*10**-7)
s0_0 = -8*10**-7/7100/gamma(-8*10**-7)
s0 = [s0_0,s1_0]
def f(s,t):
	return [t*s[1],s[0]]
t = np.arange(0,7100,(7100/len(T)))
s_e = (odeint(f,s0,t)[:,0]+odeint(f,s0,t)[:,1])*v_e
np.savetxt('electron_fluctuation.txt',s_e)
s_Fe = (odeint(f,s0,t)[:,0]+odeint(f,s0,t)[:,1])*v_Fe
np.savetxt('iron_ion_fluctuation.txt',s_Fe)
fig6 = plt.figure()
plt.plot(T,s_e)
plt.xlabel('Temperature (K)')
plt.ylabel('Fluctuation scope (cm)')
fig6.savefig('Electron_fluctuation.png')
fig7 = plt.figure()
plt.plot(T,s_Fe)
plt.xlabel('Temperature (K)')
plt.ylabel('Fluctuation scope (cm)')
fig7.savefig('Iron_fluctuation.png')
