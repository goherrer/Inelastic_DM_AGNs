'''
Code designed to find cooling times for electrons
scattering on inelastic dark matter with a vector mediator
The vector mediator now has a finite mass

For math, see notebook started by Gonzalo
'''

import matplotlib
from matplotlib import pyplot as plt
import scienceplots
#plt.style.use(["science","ieee"])
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
dataTp = np.array([]) # GeV
dataTaup = np.array([]) #sec
file = open("NGC1068Proton.csv")
for line in file:
    line = line.split(',')
    dataTp = np.append(dataTp,float(line[0]))
    dataTaup = np.append(dataTaup, float(line[1]))


rho_DM = 8e12 #GeV cm^{-3}

mp = .939 #proton mass (GeV)
Lambda = .77 #mass used for form factor (GeV)
sigma0 = 5e-35 #cm^2

tauRatioDes = 0.1

mDM_list = np.logspace(-3,6,50) #GeV
'''
delta_list = np.logspace(-3,6,50)

sigma_array = np.zeros((len(mDM_list),len(delta_list)))
sigma_array_no_ff = np.zeros((len(mDM_list),len(delta_list)))

#Ratios compared to the ground state DM mass
#fig = plt.figure()


#
#g_p = 0.1

for m in range(len(mDM_list)):
    mDM = mDM_list[m]
    print("mDM",mDM)
    for n in range(len(delta_list)):
        delta = delta_list[n]
        #print("Delta",delta)

        reducedmass = (mp*mDM)/(mp + mDM) #GeV
        
        #TeVals = np.logspace(np.log10(50),3*np.log10(2),200) #electron kinetic energies (GeV)
        TpVals = np.logspace(np.log10(3e4),np.log10(3e5),200)
        sVals = mp**2 + mDM**2 + 2* mDM * (TpVals + mp) # GeV^2
        
        kappaVals = sigma0 * rho_DM/(2 * reducedmass**2 * Kallen(mp**2, mDM**2, sVals)) #cm^{-1} GeV^{-5}
        
        tplus,tminus = tplusminus(mp,mDM,delta,sVals)
        
        TDMmin = (delta**2 - tminus)/(2*mDM)
        TDMmax = (delta**2 - tplus)/(2*mDM)
        num_TDM = 10000
        
        TDMedges = np.logspace(np.log10(TDMmin),np.log10(TDMmax),num_TDM)
        TDMvals = np.sqrt(TDMedges[:-1] * TDMedges[1:])
        dTDM = TDMedges[1:] - TDMedges[:-1]
        
        Omega = (sVals - (mDM**2 + mp**2 + delta*mDM))**2 #GeV^{4}
        
        alpha = 2* mDM**2 #GeV^2
        beta = 2*mDM**2 * delta - mDM * (delta**2 + 2 * sVals) #GeV^3
        gamma = Omega - delta * mDM * (delta**2 + 2*sVals) #GeV^4
        epsilon = delta * Omega #GeV^5
        
        qsq = 2 * mDM * TDMvals - delta**2 
        p_form_factor = (1 + qsq/Lambda**2)**(-2)
        
        integrand = (p_form_factor)**2 *(alpha*TDMvals**3 + beta * TDMvals**2 
                               + gamma* TDMvals + epsilon)*dTDM #GeV^{6}
        
        integrand_no_ff = (alpha*TDMvals**3 + beta * TDMvals**2 
                               + gamma* TDMvals + epsilon)*dTDM #GeV^{6}
        
        integral = np.sum(integrand,axis = 0)
        integral_no_ff = np.sum(integrand_no_ff, axis = 0)
        dEdt = kappaVals *integral * 3e10
        dEdt_no_ff = kappaVals *integral_no_ff * 3e10
            #GeV s^{-1}
        #Final 3e10 is multiplying by c so units are correct
        
        tau = ((1/(mp + TpVals)) * dEdt)**(-1) #s
        tau_no_ff = ((1/(mp + TpVals)) * dEdt_no_ff)**(-1) #s
        
        #print(tau[0])
        plt.plot(TpVals,tau,label = "m_{DM} = "+str(mDM))
        
        try:
            tauRatioMin = min(i for i in (tau/LogInterp(TpVals,dataTp,dataTaup)) if i>0)
            tauRatioMin_no_ff = min(i for i in (tau_no_ff/LogInterp(TpVals,dataTp,dataTaup)) if i>0)
        except:
            tauRatioMin = 0
            tauRatioMin_no_ff = 0
        sigmaDes = sigma0 * (tauRatioMin/tauRatioDes)
        sigmaDes_no_ff = sigma0 * (tauRatioMin_no_ff/tauRatioDes)
        
        sigma_array[m,n] = sigmaDes
        sigma_array_no_ff[m,n] = sigmaDes_no_ff
        
plt.legend()

#fig = plt.figure()

fig, ax = plt.subplots()
levels = np.linspace(-36,-20,9)

ex_mDM = (delta_list**2 + 2*delta_list*mp)/(2*(TpVals[-1] - delta_list))
CS = ax.contourf(mDM_list,delta_list,np.log10(np.transpose(sigma_array)),levels)

CB = fig.colorbar(CS)
CB.set_label("$\mathrm{log}_{10}(\sigma_{0}/\mathrm{cm}^2)$")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{DM}$ [GeV]")
plt.ylabel("$\delta$ [GeV]")
plt.title('Heavy Vector Mediator, NGC 1068, Proton')
#plt.plot(ex_mDM,delta_list, color = "black", linewidth = 2)

plt.xlim([mDM_list[0],mDM_list[-1]])

fig2, ax2 = plt.subplots()
levels = np.linspace(-38,-30,9)

ex_mDM = (delta_list**2 + 2*delta_list*mp)/(2*(TpVals[-1] - delta_list))
CS = ax2.contourf(mDM_list,delta_list,np.log10(np.transpose(sigma_array_no_ff)),levels)

CB2 = fig2.colorbar(CS)
CB2.set_label("$\mathrm{log}_{10}(\sigma_{0}/\mathrm{cm}^2)$")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{DM}$ [GeV]")
plt.ylabel("$\delta$ [GeV]")
plt.title('Heavy Vector Mediator, NGC 1068, Proton No Form Factor')
#plt.plot(ex_mDM,delta_list, color = "black", linewidth = 2)

plt.xlim([mDM_list[0],mDM_list[-1]])
'''

sigma_des_list = np.array([])
gDM = 1
alpha_DM = (gDM)**2/(4*pi)

mZRatio = 3
deltaRatio = 0.1
fig = plt.figure()
plt.xlabel('$T_{e}$ [GeV]')
plt.ylabel("$\\tau$ [s]")
plt.xscale('log')
plt.yscale('log')

for m in range(len(mDM_list)):
    mDM = mDM_list[m]
    print("mDM",mDM)
    delta = mDM * deltaRatio
    mZ = mDM * mZRatio

    reducedmass = (mp*mDM)/(mp + mDM) #GeV
    
    #TeVals = np.logspace(np.log10(50),3*np.log10(2),200) #electron kinetic energies (GeV)
    TpVals = np.logspace(4,7,200)#np.logspace(np.log10(3e4),np.log10(3e5),200)#np.logspace(0,np.log10(2000),200)
    sVals = mp**2 + mDM**2 + 2* mDM * (TpVals + mp) # GeV^2
    
    kappaVals = sigma0 * rho_DM/(2 * reducedmass**2 * Kallen(mp**2, mDM**2, sVals)) #cm^{-1} GeV^{-5}
    
    tplus,tminus = tplusminus(mp,mDM,delta,sVals)
    
    TDMmin = (delta**2 - tminus)/(2*mDM)
    TDMmax = (delta**2 - tplus)/(2*mDM)
    num_TDM = 10000
    dTDM = (TDMmax - TDMmin)/num_TDM
    
    TDMvals = np.linspace(TDMmin,TDMmax,num_TDM)
    
    scaling = (mZ**4)/(mZ**2 + 2*mDM * TDMvals - delta**2)**2
    
    qsq = 2 * mDM * TDMvals - delta**2 
    
    p_form_factor = (1 + qsq/Lambda**2)**(-2)
    #print('min scaling',np.min(scaling))
    #print('max scaling',np.max(scaling))
    Omega = (sVals - (mDM**2 + mp**2 + delta*mDM))**2 #GeV^{4}
    
    alpha = 2 * mDM**2 #GeV^2
    beta = 2*mDM**2 * delta - mDM * (delta**2 + 2 * sVals) #GeV^3
    gamma = Omega - delta * mDM * (delta**2 + 2*sVals) #GeV^4
    epsilon = delta * Omega #GeV^5
    
    integrand = scaling * (p_form_factor**2) *(alpha*TDMvals**3 + beta * TDMvals**2 
                           + gamma* TDMvals + epsilon)*dTDM #GeV^{6}
    
    
    
    integral = np.sum(integrand,axis = 0)
    dEdt = kappaVals *integral * 3e10
        #GeV s^{-1}
    #Final 3e10 is multiplying by c so units are correct
    
    tau = ((1/(mp + TpVals)) * dEdt)**(-1) #s
    
    #print(tau)
    
    #plt.plot(TeVals,tau,label = "m_{DM} = "+str(mDM))
    
    #print(tau[0])
    
    if m % 10 == 0:
        plt.plot(TpVals,tau,label = "mDM = 10^"+str(round(np.log10(mDM),2)))
    
    
    try:
        tauRatioMin = min(i for i in (tau/LogInterp(TpVals,dataTp,dataTaup)) if i>0)
    except:
        tauRatioMin = 0
    sigmaDes = sigma0 * (tauRatioMin/tauRatioDes)
    
    sigma_des_list = np.append(sigma_des_list,sigmaDes)
    

plt.legend()
plt.grid()


fig = plt.figure()
plt.plot(mDM_list,sigma_des_list)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{DM}$ [GeV]")
plt.ylabel("$\sigma_{0}$ [$cm^{2}$]")
plt.title("$m_{Z'} =$"+str(mZRatio) + "$m_{DM}$ ; $\delta =$"+str(deltaRatio) + "$m_{DM}$ ; $g_{DM} =$"+str(round(sqrt(alpha_DM*4*pi),3)))

plt.grid()

ylist = sigma_des_list * ((mp+mDM_list)/mp)**2 * 1/(4*(0.3)**2) \
    * (mDM_list * 5.06e13)**2 


    
fig = plt.figure()
plt.plot(mDM_list,ylist)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{DM}$ [GeV]")
plt.ylabel("$y=\epsilon^2 \\alpha_{D} (m_{DM}/m_{Z'})^4$")
plt.title("$m_{Z'} =$"+str(mZRatio) + "$m_{DM}$ ; $\delta =$"+str(deltaRatio) + "$m_{DM}$")
plt.grid()


#epsilon_list = sqrt(ylist/alpha_DM * (mZRatio)**4)
epsilon_list = sqrt(sigma_des_list * (mZRatio*mDM_list)**4
                    /(4 * 0.3**2 * alpha_DM * (mp*mDM_list/(mp+mDM_list))**2)) * 5.06e13


#plt.legend()

fig = plt.figure()
plt.plot(mDM_list * mZRatio, epsilon_list)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{Z'}$ [GeV]")
plt.ylabel("$\epsilon$")
plt.title("$m_{Z'} =$"+str(mZRatio) + "$m_{DM}$ ; $\delta =$"+str(deltaRatio) + "$m_{DM}$ ; $\\alpha_{DM} =$"+str(round(alpha_DM,3)))
plt.grid()

