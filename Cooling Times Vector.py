'''
Code designed to find cooling times for electrons
scattering on inelastic dark matter with a vector mediator

For math, see notebook started by Gonzalo
'''

import matplotlib
from matplotlib import pyplot as plt
import scienceplots
plt.style.use(["science","ieee"])
import numpy as np
from numpy import pi as pi
from numpy import sqrt, cos, sin, exp, log
from matplotlib import ticker, cm
import math
import scipy as sp
from scipy import special as spc

def Kallen(a,b,c):
    val = a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*b*c
    return(val)

def tplusminus(me,mDM,delta,s):
    '''
    Provides maximum/minimum values of Mandelstam
        variable t
    ----------
    me : Electron mass
    mDM : Dark matter mass
    delta : Dark matter mass splitting
    s : Mandelstam variable s

    Returns
    -------
    tplus, tminus: extremum values of t

    '''
    pisq = 1/(4*s) * Kallen(me**2, mDM**2, s)
    pfsq = 1/(4*s) * Kallen(me**2, (mDM+delta)**2, s)
    tplus = mDM**2 + (mDM + delta)**2 \
        - 2*(sqrt(mDM**2 + pisq)*sqrt((mDM+delta)**2 + pfsq) + sqrt(pisq*pfsq))
    tminus = mDM**2 + (mDM + delta)**2 \
        - 2*(sqrt(mDM**2 + pisq)*sqrt((mDM+delta)**2 + pfsq) - sqrt(pisq*pfsq))
        
    return(tplus,tminus)

def LogInterp(xval,xlist,ylist):
    m,b = np.polyfit(log(xlist[-5:-1]),log(ylist[-5:-1]),1)
    
    logyval = np.interp(log(xval),log(xlist),log(ylist))*np.heaviside(xlist[-1]-xval,0)\
        + (m*log(xval) + b)*np.heaviside(xval-xlist[-1],1)
    yval = exp(logyval)
    return(yval)
    

#ImportData
dataTe = np.array([]) # GeV
dataTaue = np.array([]) #sec
file = open("ElectronCoolingTXSdata.csv")
for line in file:
    line = line.split(',')
    dataTe = np.append(dataTe,float(line[0]))
    dataTaue = np.append(dataTaue, float(line[1]))


rho_DM = 8e12 #GeV cm^{-3}

me = 511*1e-6 #electron mass (GeV)
sigma0 = 5e-35 #cm^2

tauRatioDes = 0.1

mDM_list = np.logspace(-6,6,200) #GeV
delta_list = np.logspace(-5,6,150) #GeV

sigma_array = np.zeros((len(mDM_list),len(delta_list)))

fig = plt.figure()
plt.xlabel('$T_{e}$ [GeV]')
plt.ylabel("$\\tau$ [s]")
plt.xscale('log')
plt.yscale('log')


for m in range(len(mDM_list)):
    mDM = mDM_list[m]
    #print("mDM",mDM)
    for n in range(len(delta_list)):
        delta = delta_list[n]
        #print("Delta",delta)

        reducedmass = (me*mDM)/(me + mDM) #GeV
        
        #TeVals = np.logspace(np.log10(50),3*np.log10(2),200) #electron kinetic energies (GeV)
        TeVals = np.logspace(0,3,200)#np.logspace(np.log10(50),np.log10(2000),200)
        sVals = me**2 + mDM**2 + 2* mDM * (TeVals + me) # GeV^2
        
        kappaVals = sigma0 * rho_DM/(2 * reducedmass**2 * Kallen(me**2, mDM**2, sVals)) #cm^{-1} GeV^{-5}
        
        tplus,tminus = tplusminus(me,mDM,delta,sVals)
        
        TDMmin = (delta**2 - tminus)/(2*mDM)
        TDMmax = (delta**2 - tplus)/(2*mDM)
        
        Omega = (sVals - (mDM**2 + me**2 + delta*mDM))**2 #GeV^{4}
        
        alpha = 2* mDM**2 #GeV^2
        beta = 2*mDM**2 * delta - mDM * (delta**2 + 2 * sVals) #GeV^3
        gamma = Omega - delta * mDM * (delta**2 + 2*sVals) #GeV^4
        epsilon = delta * Omega #GeV^5
        
        dEdt = kappaVals \
            * (alpha/4 * (TDMmax**4 - TDMmin**4) + beta/3 * (TDMmax**3 - TDMmin**3)
              + gamma/2 * (TDMmax**2 - TDMmin**2) + epsilon * (TDMmax - TDMmin)) * 3e10
            #GeV s^{-1}
        #Final 3e10 is multiplying by c so units are correct
        
        tau = ((1/(me + TeVals)) * dEdt)**(-1) #s
        
        #print(tau[0])
        #plt.plot(TeVals,tau,label = "m_{DM} = "+str(mDM))
        
        try:
            tauRatioMin = min(i for i in (tau/LogInterp(TeVals,dataTe,dataTaue)) if i>0)
        except:
            tauRatioMin = 0
        sigmaDes = sigma0 * (tauRatioMin/tauRatioDes)
        
        sigma_array[m,n] = sigmaDes

plt.legend()

#fig = plt.figure()

fig, ax = plt.subplots()
levels = np.linspace(-40,-28,9)

ex_mDM = (delta_list**2 + 2*delta_list*me)/(2*(TeVals[-1] - delta_list))
CS = ax.contourf(mDM_list,delta_list,np.log10(np.transpose(sigma_array)),levels)

CB = fig.colorbar(CS)
CB.set_label("$\mathrm{log}_{10}(\sigma_{e0}/\mathrm{cm}^2)$")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{DM}$ [GeV]")
plt.ylabel("$\delta$ [GeV]")
#plt.title('Vector Mediator')
#plt.plot(ex_mDM,delta_list, color = "black", linewidth = 2)

plt.xlim([mDM_list[0],mDM_list[-1]])

'''
vrelDD = 1e-3
deltaDD = -me -mDM_list + sqrt(me**2 + mDM_list**2 + 2*me*mDM_list/sqrt(1-vrelDD**2))
plt.plot(mDM_list,deltaDD,label = "Direct Detection")
'''

vrelWD = .8
deltaWD = - me - mDM_list + sqrt(me**2 +mDM_list**2 + 2*me*mDM_list/sqrt(1-vrelWD**2))
plt.plot(mDM_list, deltaWD,label = "White Dwarf")

plt.fill_between([1e3,mDM_list[-1]],delta_list[0],delta_list[-1],color = "gray",alpha = 0.5,
                 label = "Electroweak Dark Matter")

plt.legend(fontsize = 5, loc = "upper left")