'''
Code designed to find cooling times for electrons
scattering on inelastic dark matter with a vector mediator
The vector mediator now has a finite mass

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

deltaEx = 1 
mDMEx = 1000 

me = 1 #electron mass (GeV)
T = 1e6
s = mDMEx**2 + me**2 + 2*mDMEx * (me+T)
tplusEx,tminusEx = tplusminus(me,mDMEx,deltaEx,s)
print('min q^2',(deltaEx**2 - tminusEx)/(2*mDMEx))

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

mDM_list = np.logspace(-3,2,60) #GeV

sigma_des_list = np.array([])
'''
fig = plt.figure()
plt.xlabel('$T_{e}$ [GeV]')
plt.ylabel("$\\tau$ [s]")
plt.xscale('log')
plt.yscale('log')
'''
#Ratios compared to the ground state DM mass
#fig = plt.figure()


#========================================
# Change Parameters in Paragraph Below
#=======================================

mZRatio = 0.01
deltaRatio = 0.1
gDM = sqrt(4*pi)
alpha_DM = (gDM)**2/(4*pi)
g_e = 1 

fig = plt.figure()
plt.xlabel('$T_{e}$ [GeV]')
plt.ylabel("$\\tau$ [s]")
plt.xscale('log')
plt.yscale('log')

print(sigma0)

for m in range(len(mDM_list)):
    mDM = mDM_list[m]
    print("mDM",mDM)
    delta = mDM * deltaRatio
    mZ = mDM * mZRatio
    
    muDMe = (mDM*me)/(mDM+me)
    

    reducedmass = (me*mDM)/(me + mDM) #GeV
    
    #TeVals = np.logspace(np.log10(50),3*np.log10(2),200) #electron kinetic energies (GeV)
    TeVals = np.logspace(0,np.log10(2000),200)
    sVals = me**2 + mDM**2 + 2* mDM * (TeVals + me) # GeV^2
    
    kappaVals = sigma0 * rho_DM/(2 * reducedmass**2 * Kallen(me**2, mDM**2, sVals)) #cm^{-1} GeV^{-5}
    
    tplus,tminus = tplusminus(me,mDM,delta,sVals)
    
    TDMmin = (delta**2 - tminus)/(2*mDM)
    TDMmax = (delta**2 - tplus)/(2*mDM)
    num_TDM = 10000
    dTDM = (TDMmax - TDMmin)/num_TDM
    
    TDMvals = np.linspace(TDMmin,TDMmax,num_TDM)
    
    scaling = (mZ**4)/(mZ**2 + 2*mDM * TDMvals - delta**2)**2
    #print('min scaling',np.min(scaling))
    #print('max scaling',np.max(scaling))
    Omega = (sVals - (mDM**2 + me**2 + delta*mDM))**2 #GeV^{4}
    
    alpha = 2 * mDM**2 #GeV^2
    beta = 2*mDM**2 * delta - mDM * (delta**2 + 2 * sVals) #GeV^3
    gamma = Omega - delta * mDM * (delta**2 + 2*sVals) #GeV^4
    epsilon = delta * Omega #GeV^5
    
    integrand = scaling * (alpha*TDMvals**3 + beta * TDMvals**2 
                           + gamma* TDMvals + epsilon)*dTDM #GeV^{6}
    
    dsigmadTExplicit = sigma0*scaling* (mDM * (sVals - (mDM**2 +me**2 + delta*mDM))**2 + mDM * TDMvals * (2*mDM*TDMvals - delta**2 - 2*sVals))\
        /(2*muDMe**2 * Kallen(mDM**2,me**2,sVals))
    
    '''
    fig = plt.figure()
    plt.plot(TDMvals,integrand[:,0]/((TDMvals[:,0] + delta)*dTDM) * kappaVals * mDM/rho_DM)
    #plt.plot(TDMvals,integrand[:,0]/(dTDM) * kappaVals * mDM/rho_DM)
    plt.plot(TDMvals,dsigmadTExplicit)
    plt.ylabel('dsigmadT [cm2 GeV-1]')
    plt.xlabel('T_DM [GeV]')
    plt.xscale('log')
    plt.yscale('log')
    '''
    
    
    integral = np.sum(integrand,axis = 0)
    dEdt = kappaVals *integral * 3e10
        #GeV s^{-1}
    #Final 3e10 is multiplying by c so units are correct
    
    tau = ((1/(me + TeVals)) * dEdt)**(-1) #s
    
    #print(tau)
    
    #plt.plot(TeVals,tau,label = "m_{DM} = "+str(mDM))
    
    #print(tau[0])
    
    if m % 10 == 0:
        plt.plot(TeVals,tau,label = "mDM = "+str(round(mDM,2)))
    
    
    try:
        tauRatioMin = min(i for i in (tau/LogInterp(TeVals,dataTe,dataTaue)) if i>0)
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

ylist = sigma_des_list * ((me+mDM_list)/me)**2 * 1/(4*(0.3)**2) \
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
                    /(4 * 0.3**2 * alpha_DM * (me*mDM_list/(me+mDM_list))**2)) * 5.06e13


#plt.legend()

fig = plt.figure()
plt.plot(mDM_list * mZRatio, epsilon_list)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{Z'}$ [GeV]")
plt.ylabel("$\epsilon$")
plt.title("$m_{Z'} =$"+str(mZRatio) + "$m_{DM}$ ; $\delta =$"+str(deltaRatio) + "$m_{DM}$ ; $\\alpha_{DM} =$"+str(round(alpha_DM,3)))
plt.grid()
