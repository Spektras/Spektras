import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy
import scipy.interpolate as interpol
t = 120 # integration time in seconds, done separately for each meteoric event
epsilon = 8.854e-012
h = 6.626e-034
c = 3e008
e = 1.602e-019
i = scipy.sqrt(-1)
lambda_de = np.loadtxt('lambda_De.txt')
d = 1+np.gradient(lambda_de)
d = np.asarray([d,d,d])
lambda_de = np.asarray([lambda_de,lambda_de,lambda_de])
lambda_m = np.loadtxt('lambda_M.txt')
lambda_m = np.asarray([lambda_m,lambda_m,lambda_m])
d_omega = np.loadtxt('critical_frequence_change.txt')
I = np.loadtxt('meteoric_intensity.txt')
omega1 = d_omega+23400 #initial ep frequency in s-1
omega1 = np.asarray([omega1,omega1,omega1])
omega2 = omega1 + c/lambda_m
omega2 = np.asarray([omega2,omega2,omega2])
M_linear = np.asarray([(lambda_de),(lambda_de+d),(lambda_de-d)])
M_quadratic = np.asarray([(lambda_de)**2,(lambda_de+d)**2,(lambda_de-d)**2])
M_cubic = np.asarray([(lambda_de)**3,(lambda_de+d)**3,(lambda_de-d)**3])
M = np.asarray([M_linear,M_quadratic,M_cubic])
E1 = e**2*(1/(2*np.pi*epsilon))*((lambda_de+d)**2-lambda_de**2)/(lambda_de**2*(lambda_de+d)**2)*(h*np.pi+4*np.pi**2*omega1*i)/(np.pi*(1/3+omega1*t*i))*t**3*np.pi*M**3*(1/3+omega1*t*i)*np.exp(-t**3*np.pi*M**3*(1/3+omega1*t*i)*2*np.pi*t*i)*(2/(M**2*t**4)+5/(6*M**2*i*omega1*t)+2/(M**5*np.pi*i*t**7))
E1[E1>1e07]=0
E2 = e**2*(1/(2*np.pi*epsilon))*((lambda_de+d)**2-lambda_de**2)/(lambda_de**2*(lambda_de+d)**2)*(h*np.pi+4*np.pi**2*omega2*i)/(np.pi*(1/3+omega2*t*i))*t**3*np.pi*M**3*(1/3+omega2*t*i)*np.exp(-t**3*np.pi*M**3*(1/3+omega2*t*i)*2*np.pi*t*i)*(2/(M**2*t**4)+5/(6*M**2*i*omega2*t)+2/(M**5*np.pi*i*t**7))
E2[np.isinf(E2)]=0
E2 = np.nan_to_num(E2)
d_E = E2-E1
d_E[d_E<0]=0
lambda1 = 1000*h*c/(np.abs(E1.imag))
lambda1[np.isinf(lambda1)]=0
lambda1 = np.nan_to_num(lambda1)
lambda2 = 1000*h*c/(np.abs(E2.imag))
lambda2[np.isinf(lambda2)]=0
lambda2 = np.nan_to_num(lambda2)
lambda_m = h*c/(np.abs(d_E.imag))
lambda_m[np.isinf(lambda_m)]=0
d_lambda = 1000*np.abs(lambda2-lambda1)
# all units set up onto basic SI
I = np.asarray([I,I,I])
energy = [(E1[1][0][0]),(E2[1][0][0])]
wavelength = [(lambda1[1][0][0]),(lambda2[1][0][0]),(lambda_m[1][0][0]),(I[1])]
np.savetxt('energy.txt',energy)
np.savetxt('wavelength.txt',wavelength)
lambda_mm = lambda2[1][0][0]
I_mm = I[1]
f = interpol.interp1d(lambda_mm,I_mm)
F = f(lambda_mm)
F[np.isinf(F)]=0
F = np.nan_to_num(F)
fig1 = plt.figure()
plt.plot(F)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
fig1.savefig('meteoric_spectra.jpg')