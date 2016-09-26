import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.misc
import math
from scipy.misc import factorial
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.special import gamma,airy
ablation = np.loadtxt('ablace.spa')
lambda_m_nm = ablation[:,0]
lambda_m = lambda_m_nm/10**9
I = ablation[:,1]
def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return array[idx]
data = np.loadtxt('data_200_800.txt')
zeros = np.zeros(len(lambda_m_nm)-len(data))
lambda0 = np.insert((data[:,0]),zeros,0)
gi = data[:,1]
gj = data[:,2]
E_down = data[:,3]
E_upp = data[:,4]
Aij = data[:,5]
position = np.where(lambda_m_nm>find_nearest(lambda_m_nm,lambda0))
np.savetxt('position.txt',position)
position = np.loadtxt('position.txt')
correct = np.ones((len(lambda_m_nm)-len(position)))
position = np.insert(position,correct,0)
index = np.asarray(position,dtype='int')
lambda_lines = lambda0/10**9
I_lines = I[index] + np.abs(min(I[index]))
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
F = 1 #needs to corrected
def mean(x):
	return np.sum(x)/len(x)
T0 = (wien/lambda_m[index]-mean(wien/lambda_m[index])+np.abs(min(wien/lambda_m[index]-mean(wien/lambda_m[index])))) - (np.abs(wien/lambda_m[index]-mean(wien/lambda_m[index]))+(max(np.abs(wien/lambda_m[index]-mean(wien/lambda_m[index])))-np.abs(wien/lambda_m[index]-mean(wien/lambda_m[index]))))+np.abs(min((wien/lambda_m[index]-mean(wien/lambda_m[index])+np.abs(min(wien/lambda_m[index]-mean(wien/lambda_m[index])))) - (np.abs(wien/lambda_m[index]-mean(wien/lambda_m[index]))+(max(np.abs(wien/lambda_m[index]-mean(wien/lambda_m[index])))-np.abs(wien/lambda_m[index]-mean(wien/lambda_m[index]))))))
E = h*c/lambda_m[index]
a = k1**(k1-1)/factorial(k1)*(1/((T_min*(E*k_B)**(T_min))-np.log(T_min)))**k1
T0_1 = np.abs(np.log(1/(a+np.e)))
b = T_min/(np.log(T_min-I_lines*E*k_B))
T1_1 = np.abs(T0_1*b)*scipy.misc.logsumexp(I_lines)
T = (np.abs(np.gradient(np.gradient(T1_1)))+T0)**2/(2*T0)
dT = np.gradient(T)
v_e = 4.19*10**7*T**1/2 #cm/s
v_Fe = 9.79*10**5*(9.2703e-026/m_p)**1/2*T**1/2 # mean q velocity of iron ion
s1_0 = 2*10**-7/7100/gamma(2*10**-7)
s0_0 = -8*10**-7/7100/gamma(-8*10**-7)
s0 = [s0_0,s1_0]
def f(s,t):
	return [t*s[1],s[0]]
t = np.arange(0,7100,(7100/len(T)))
s_e = (odeint(f,s0,t)[:,0]+odeint(f,s0,t)[:,1])*v_e
s_Fe = (odeint(f,s0,t)[:,0]+odeint(f,s0,t)[:,1])*v_Fe
fig0 = plt.figure()
plt.plot(T,s_e)
plt.xlabel('Temperature (K)')
plt.ylabel('Fluctuation scope (cm)')
fig0.savefig('Electron_fluctuation.png')
fig1 = plt.figure()
plt.plot(T,s_Fe)
plt.xlabel('Temperature (K)')
plt.ylabel('Fluctuation scope (cm)')
fig1.savefig('Iron_fluctuation.png')
lambda1_lines = lambda_lines*10**9
np.savetxt('temperature.txt',T)
fig2 = plt.figure()
plt.plot(lambda_m_nm[index],T)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Temperature (K)')
plt.xlim(xmin=190)
plt.xlim(xmax=900)
fig2.savefig('Temperature_fit0.png')
fig3 = plt.figure()
plt.plot(T,I_lines)
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Temperature (K)')
fig3.savefig('Temperature_fit.png')
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
geometry = [r_min, r_max, dN]
master1 = [lambda_m_nm, I, T]
np.savetxt('master1.txt',master1)
lambda1 = lambda_lines[(len(lambda_lines)-len(gi)):len(lambda_lines)]
I = I_lines[(len(I_lines)-len(gi)):len(I_lines)]
sum_g = np.sum(gj)
T_lines = T[(len(T)-len(gi)):len(T)]
T_lines[np.isinf(T_lines)]=0
T_lines = np.nan_to_num(T_lines)
Q_s = sum_g*np.exp(-(E_upp*e/8065.54)/(k_B*T_lines))
stat = np.exp(-(E_upp*e/8065.54)/(k_B*T_lines))
nu_ij = c/lambda1
C = np.sqrt(6.022e026*((4*np.pi*I*Q_s)/(F*Aij*gi*nu_ij*h*stat))*(odeint(f,s0,t)[len(lambda_lines)-len(gi):len(lambda_lines),0]+odeint(f,s0,t)[len(lambda_lines)-len(gi):len(lambda_lines),1])**4)
C[np.isinf(C)]=0
C = np.nan_to_num(C)
C_C = np.sum(C[1:1130])
C_Na = np.sum(C[1131:3172])
C_Mg = np.sum(C[3173:4452])
C_Al = np.sum(C[4453:5698])
C_Si = np.sum(C[5698:7100])
C_P = np.sum(C[7101:7416])
C_S = np.sum(C[7417:9635])
C_Cl = np.sum(C[9636:10212])
C_K = np.sum(C[10213:10576])
C_Ca = np.sum(C[10577:11210])
C_Ti = np.sum(C[11211:16643])
C_V = np.sum(C[16644:18214])
C_Cr = np.sum(C[18215:26822])
C_Mn = np.sum(C[26823:31063])
C_Fe = np.sum(C[31064:47418])
C_Co = np.sum(C[47419:50887])
C_Ni = np.sum(C[50888:51755])
C_Cu = np.sum(C[51756:52982])
C_Zn = np.sum(C[52983:53097])
C_Se = np.sum(C[53098:53183])
C_Sr = np.sum(C[53184:54465])
C_Mo = np.sum(C[54466:55856])
C_Cd = np.sum(C[55857:56035])
C_Pb = np.sum(C[56036:56246])
C_U = np.sum(C[56247:56729])
C_Be = np.sum(C[56730:57620])
C_sum = np.sum(C)
def pm(x):
	return np.sqrt(x-mean(x))**2/(len(x)*(len(x)-1))
pm = pm(C)
pm[np.isinf(pm)]=0
pm = np.nan_to_num(pm)
pm_C_C = np.sum(pm[1:1130])
pm_C_Na = np.sum(pm[1131:3172])
pm_C_Mg = np.sum(pm[3173:4452])
pm_C_Al = np.sum(pm[4453:5698])
pm_C_Si = np.sum(pm[5698:7100])
pm_C_P = np.sum(pm[7101:7416])
pm_C_S = np.sum(pm[7417:9635])
pm_C_Cl = np.sum(pm[9636:10212])
pm_C_K = np.sum(pm[10213:10576])
pm_C_Ca = np.sum(pm[10577:11210])
pm_C_Ti = np.sum(pm[11211:16643])
pm_C_V = np.sum(pm[16644:18214])
pm_C_Cr = np.sum(pm[18215:26822])
pm_C_Mn = np.sum(pm[26823:31063])
pm_C_Fe = np.sum(pm[31064:47418])
pm_C_Co = np.sum(pm[47419:50887])
pm_C_Ni = np.sum(pm[50888:51755])
pm_C_Cu = np.sum(pm[51756:52982])
pm_C_Zn = np.sum(pm[52983:53097])
pm_C_Se = np.sum(pm[53098:53183])
pm_C_Sr = np.sum(pm[53184:54465])
pm_C_Mo = np.sum(pm[54466:55856])
pm_C_Cd = np.sum(pm[55857:56035])
pm_C_Pb = np.sum(pm[56036:56246])
pm_C_U = np.sum(pm[56247:56729])
pm_C_Be = np.sum(pm[56730:56720])
pm_sum = np.sum(pm)
def dispersion(x):
	return (x-mean(x))**2/(len(x)*(len(x)-1))
dispersion = dispersion(C)
dispersion[np.isinf(dispersion)]=0
dispersion = np.nan_to_num(dispersion)
dispersion_C_C = np.sum(dispersion[1:1130])
dispersion_C_Na = np.sum(dispersion[1131:3172])
dispersion_C_Mg = np.sum(dispersion[3173:4452])
dispersion_C_Al = np.sum(dispersion[4453:5698])
dispersion_C_Si = np.sum(dispersion[5698:7100])
dispersion_C_P = np.sum(dispersion[7101:7416])
dispersion_C_S = np.sum(dispersion[7417:9635])
dispersion_C_Cl = np.sum(dispersion[9636:10212])
dispersion_C_K = np.sum(dispersion[10213:10576])
dispersion_C_Ca = np.sum(dispersion[10577:11210])
dispersion_C_Ti = np.sum(dispersion[11211:16643])
dispersion_C_V = np.sum(dispersion[16644:18214])
dispersion_C_Cr = np.sum(dispersion[18215:26822])
dispersion_C_Mn = np.sum(dispersion[26823:31063])
dispersion_C_Fe = np.sum(dispersion[31064:47418])
dispersion_C_Co = np.sum(dispersion[47419:50887])
dispersion_C_Ni = np.sum(dispersion[50888:51755])
dispersion_C_Cu = np.sum(dispersion[51756:52982])
dispersion_C_Zn = np.sum(dispersion[52983:53097])
dispersion_C_Se = np.sum(dispersion[53098:53183])
dispersion_C_Sr = np.sum(dispersion[53184:54465])
dispersion_C_Mo = np.sum(dispersion[54466:55856])
dispersion_C_Cd = np.sum(dispersion[55857:56035])
dispersion_C_Pb = np.sum(dispersion[56036:56246])
dispersion_C_U = np.sum(dispersion[56247:56729])
dispersion_C_Be = np.sum(dispersion[56730:56720])
dispersion_sum = np.sum(dispersion)
C_C = np.sum(C[1:1130])**2/dispersion_C_C*len(C[1:1130])
C_Na = np.sum(C[1131:3172])**2/dispersion_C_Na*len(C[1131:3172])
C_Mg = np.sum(C[3173:4452])**2/dispersion_C_Mg*len(C[3173:4452])
C_Al = np.sum(C[4453:5698])**2/dispersion_C_Al*len(C[4453:5698])
C_Si = np.sum(C[5699:7100])**2/dispersion_C_Si*len(C[5699:7100])
C_P = np.sum(C[7101:7416])**2/dispersion_C_P*len(C[7101:7416])
C_S = np.sum(C[7417:9635])**2/dispersion_C_S*len(C[7417:9635])
C_Cl = np.sum(C[9636:10212])**2/dispersion_C_Cl*len(C[9636:10212])
C_K = np.sum(C[10213:10576])**2/dispersion_C_K*len(C[10213:10576])
C_Ca = np.sum(C[10577:11210])**2/dispersion_C_Ca*len(C[10577:11210])
C_Ti = np.sum(C[11211:16643])**2/dispersion_C_Ti*len(C[11211:16643])
C_V = np.sum(C[16644:18214])**2/dispersion_C_V*len(C[16644:18214])
C_Cr = np.sum(C[18215:26822])**2/dispersion_C_Cr*len(C[18215:26822])
C_Mn = np.sum(C[26823:31063])**2/dispersion_C_Mn*len(C[26823:31063])
C_Fe = np.sum(C[31064:47418])**2/dispersion_C_Fe*len(C[31063:47418])
C_Co = np.sum(C[47419:50887])**2/dispersion_C_Co*len(C[47419:50887])
C_Ni = np.sum(C[50888:51755])**2/dispersion_C_Ni*len(C[50888:51755])
C_Cu = np.sum(C[51756:52982])**2/dispersion_C_Cu*len(C[51756:52982])
C_Zn = np.sum(C[52983:53097])**2/dispersion_C_Zn*len(C[52983:53097])
C_Se = np.sum(C[53098:53183])**2/dispersion_C_Se*len(C[53098:53183])
C_Sr = np.sum(C[53184:54465])**2/dispersion_C_Sr*len(C[53184:54465])
C_Mo = np.sum(C[54466:55856])**2/dispersion_C_Mo*len(C[54466:55856])
C_Cd = np.sum(C[55857:56035])**2/dispersion_C_Cd*len(C[55857:56035])
C_Pb = np.sum(C[56036:56246])**2/dispersion_C_Pb*len(C[56036:56246])
C_U = np.sum(C[56247:56729])**2/dispersion_C_U*len(C[56247:56729])
C_Be = np.sum(C[56730:57620])**2/dispersion_C_Be*len(C[56730:56720])
C_sum = C_C+C_Na+C_Mg+C_Al+C_Si+C_P+C_S+C_Cl+C_K+C_Ca+C_Ti+C_V+C_Cr+C_Mn+C_Fe+C_Co+C_Ni+C_Cu+C_Zn+C_Se+C_Sr+C_Mo+C_Cd+C_Pb+C_U
concentration = [C_C, C_Na, C_Mg, C_Al, C_Si, C_P, C_S, C_Cl, C_K, C_Ca, C_Ti, C_V, C_Cr, C_Mn, C_Fe, C_Co, C_Ni, C_Cu, C_Zn, C_Se, C_Sr, C_Mo, C_Cd, C_Pb, C_U, C_Be, C_sum]
concentration_pm = [pm_C_C, pm_C_Na, pm_C_Mg, pm_C_Al, pm_C_Si, pm_C_P, pm_C_S, pm_C_Cl, pm_C_K, pm_C_Ca, pm_C_Ti, pm_C_V, pm_C_Cr, pm_C_Mn, pm_C_Fe, pm_C_Co, pm_C_Ni, pm_C_Cu, pm_C_Zn, pm_C_Se, pm_C_Sr, pm_C_Mo, pm_C_Cd, pm_C_Pb, pm_C_U, pm_C_Be, pm_sum]
concentration_dispersion = [dispersion_C_C, dispersion_C_Na, dispersion_C_Mg, dispersion_C_Al, dispersion_C_Si, dispersion_C_P, dispersion_C_S, dispersion_C_Cl, dispersion_C_K, dispersion_C_Ca, dispersion_C_Ti, dispersion_C_V, dispersion_C_Cr, dispersion_C_Mn, dispersion_C_Fe, dispersion_C_Co, dispersion_C_Ni, dispersion_C_Cu, dispersion_C_Zn, dispersion_C_Se, dispersion_C_Sr, dispersion_C_Mo, dispersion_C_Cd, dispersion_C_Pb, dispersion_C_U, dispersion_C_Be, dispersion_sum]
concentration_relative = [C_C/C_sum,C_Na/C_sum,C_Mg/C_sum,C_Al/C_sum,C_Si/C_sum,C_P/C_sum,C_S/C_sum,C_Cl/C_sum,C_K/C_sum,C_Ca/C_sum,C_Ti/C_sum,C_V/C_sum,C_Cr/C_sum,C_Mn/C_sum,C_Fe/C_sum,C_Co/C_sum,C_Ni/C_sum,C_Cu/C_sum,C_Zn/C_sum,C_Se/C_sum,C_Sr/C_sum,C_Mo/C_sum,C_Cd/C_sum,C_Pb/C_sum,C_U/C_sum,C_Be/C_sum,1]
concentration_sigma = np.sqrt((np.asarray(concentration_pm)**2+np.asarray(concentration_dispersion))/2)
concentration_ratio = [C_Fe/C_Mg,C_Na/C_Mg,C_Na/C_Fe,C_Fe/C_Si,C_K,C_Ca,C_Fe/C_Ni,C_Fe/C_Co,C_Si/C_Al]
np.savetxt('concentration.txt',concentration)
np.savetxt('concetration_statistics.txt',concentration_sigma)
np.savetxt('concentration_relative.txt',concentration_relative)
np.savetxt('concentration_ratio.txt',concentration_ratio)
