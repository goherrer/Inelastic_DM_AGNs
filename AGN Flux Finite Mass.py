'''
Calculating the resulting chi distribution from
    scattering with a finite mass mediator
'''

import matplotlib
from matplotlib import pyplot as plt
import scienceplots
plt.style.use(["science","ieee","bright"])
import numpy as np
from numpy import pi as pi
from numpy import sqrt, cos, sin, exp, log
from matplotlib import ticker, cm
import math
import scipy as sp
from scipy import special as spc
#from Self_PDF import uPDFfunc, dPDFfunc, ubarPDFfunc,dbarPDFfunc, Qmin
from numpy.random import randint
import numpy.random as rand
import Colinear_Flux as CF

def sigma0valsFiniteMass(gp,gChi,mZ,mp,mDM,delta):
    '''
    Code to determine non-relativistic cross section from
    model parameters

    Parameters
    ----------
    gp : Proton coupling
    gChi : Dark Matter Coupling
    mZ: Mediator Mass [GeV]
    mp : Mass of proton [GeV]
    mDM : Mass of dark matter ground state [GeV]
    delta : Dark matter mass splitting [GeV]

    Returns
    -------
    sigmaSM0 : Non-relativistic proton-DM scattering [cm^2]
    sigma10 : Non-relativisitic chi_1 + chi_1 -> chi_2 + chi_2 scattering [cm^2]
    sigma20 : Non-relativisitic chi_1 + chi_2 -> chi_1 + chi_2 scattering [cm^2]
    '''
    mu_pDM = (mp * mDM)/(mp + mDM) #GeV
    mu_11 = (mDM * mDM) / (mDM + mDM) #GeV
    mu_12 = ((mDM + delta) * mDM) / (2*mDM + delta)
    
    sigmaSM0GeV = (gp*gChi/mZ**2)**2 * mu_pDM**2 /pi #GeV^{-2}
    sigma10GeV =  (gChi/mZ)**4 * mu_11**2 /pi #GeV^{-2}
    sigma20GeV =  (gChi/mZ)**4 * mu_12**2 /pi #GeV^{-2}
    
    sigmaSM0 = sigmaSM0GeV * (1/5.06e13)**2 #cm^2
    sigma10 = sigma10GeV * (1/5.06e13)**2 #cm^2
    sigma20 = sigma20GeV * (1/5.06e13)**2 #cm^2
    
    return(sigmaSM0,sigma10,sigma20)

def dsigmadE2SMFiniteMass(sigma0SM, mZ, Ei, E2, mSM, mDM, delta,max_angle = pi):
    '''
    Function for determining the differential cross section
        for SM particles scattering with DM

    Inputs:
        sigma0SM: Cross section of non-relativistic scattering [cm^2]
        mZ : Mediator Mass [GeV]
        Ei: Energy of incoming SM particle [GeV]
        E2: Energy of the outgoing chi_{2} particle [GeV]
        mSM: Mass of the SM particle [GeV]
        mDM: Mass of the chi_{1} particle [GeV]
        delta: Mass splitting of the chi particles [GeV]

    Output:
        dsdE2: Differential Energy Cross Section [cm^{2} GeV^{-1}]
    '''
    mp = 0.94
    s = mSM**2 + mDM**2 + 2*mDM*Ei
    
    tplus,tminus = CF.tplusminus(mSM,mDM,delta,s)
    TDMmin = (delta**2 - tminus)/(2*mDM) * np.heaviside(s - (mSM+mDM+delta)**2,0)
    TDMmax = (delta**2 - tplus)/(2*mDM) * np.heaviside(s - (mSM+mDM+delta)**2,0)
    
    #print('TDMmin', TDMmin)
    #print('TDMmax', TDMmax)
    TSM = Ei - mSM
    TDM = E2 - mDM
    qsq = 2*mDM * TDM - delta**2
    reducedmass = mDM*mSM/(mDM + mSM)
    
    Lambda = 0.77 #GeV
    
    xsq = qsq/Lambda**2
    FormFactor = (1+xsq)**(-2)

    numerator = mDM * ((s - (mDM**2 + mSM**2 + delta*mDM))**2 + mDM*TDM * (qsq - 2*s)) * FormFactor**2
    denominator = 2 * reducedmass**2 * CF.Kallen(s, mDM**2, mSM**2)
    
    Med_FormFactor = (mZ**2/(mZ**2 + qsq))

    dsdE2 = sigma0SM * (Med_FormFactor)**2 * numerator/denominator * np.heaviside(TDM - TDMmin,0)*np.heaviside(TDMmax - TDM,0)

    #Angular Requirements
    
    cos_theta_num = -delta**2 + 2*mSM*(delta+TDM) + 2*mDM*TDM + 2*delta*TSM + 2*TSM*TDM
    cos_theta_den_sq = 4*(TSM*(2*mSM+TSM))*(TDM*(2*delta + 2*mDM + TDM))
    cos_theta = cos_theta_num/sqrt(np.abs(cos_theta_den_sq)) * np.heaviside(cos_theta_den_sq,0)
    
    angular_req = np.heaviside(cos_theta - cos(max_angle),1)
    
    #Calculate Deep Inelastic Scattering
    #Just use up and down quarks
    #Assume the coupling to quarks is the same as protons
    mq = 0.002 #GeV
    reducedmassq = mDM*mq/(mDM + mq)
    sigma0q = sigma0SM  * (reducedmassq / reducedmass)**2 #*(1/3)**2
    BjxVals = np.linspace(0.1,1,100)
    dx = BjxVals[1] - BjxVals[0]
    
    dsdE2DIS = np.zeros(dsdE2.shape)
    
    '''
    for Bjx in BjxVals:
        
        #print(np.size(qsq))
        fxvals = (uPDFfunc(Bjx,sqrt(np.abs(qsq))) + dPDFfunc(Bjx,sqrt(np.abs(qsq))))\
            * np.transpose([np.heaviside(qsq,0)])
        #print(np.size(fxvals))
        sq = s*Bjx
        tplusq,tminusq = CF.tplusminus(mq,mDM,delta,sq)
        TDMminq = (delta**2 - tminusq)/(2*mDM) * np.heaviside(sq - (mq+mDM+delta)**2,0)
        TDMmaxq = (delta**2 - tplusq)/(2*mDM) * np.heaviside(sq - (mq+mDM+delta)**2,0)
        
        #There is no good reason for this choice here. I need a better threshold
        DISThreshold2 = np.heaviside(sq - 10*mDM**2 - 10*mp**2,0)
        DISThreshold = np.heaviside(qsq - Qmin**2, 0)
        
        numeratorq = mDM * ((sq - (mDM**2 + mq**2 + delta*mDM))**2 + mDM*TDM * (qsq - 2*sq))\
            * (1-FormFactor**2) * (Med_FormFactor**2)
    
        denominatorq = 2 * reducedmassq**2 * CF.Kallen(sq, mDM**2, mq**2)
        
        #print(np.product(np.isfinite(fxvals)))
        try:
            try:
                dsdE2DIS += sigma0q*fxvals*dx * numeratorq/denominatorq \
                    * np.heaviside(TDM - TDMminq,0)*np.heaviside(TDMmaxq - TDM,0)*DISThreshold*DISThreshold2
            except:
                fxvals = np.transpose(fxvals)
                dsdE2DIS += sigma0q*fxvals*dx * numeratorq/denominatorq \
                    * np.heaviside(TDM - TDMminq,0)*np.heaviside(TDMmaxq - TDM,0)\
                        *DISThreshold*DISThreshold2
        except:
            try:
                dsdE2DIS += (sigma0q*fxvals*dx * numeratorq/denominatorq \
                    * np.heaviside(TDM - TDMminq,0)*np.heaviside(TDMmaxq - TDM,0)*DISThreshold*DISThreshold2)[0]
            except:
                fxvals = np.transpose(fxvals)
                dsdE2DIS += (sigma0q*fxvals*dx * numeratorq/denominatorq \
                    * np.heaviside(TDM - TDMminq,0)*np.heaviside(TDMmaxq - TDM,0)\
                        *DISThreshold*DISThreshold2)[0]

    '''
    return ((dsdE2 + dsdE2DIS) * angular_req)

def SingleScatterFluxFiniteMassNoDec(mDM,delta,gp,gChi,mZ, 
                 dist = 1835.4 * (3.86e24), MBH = 1e8, Gamma_B = 20, alpha_p = 2, cp = 2.7e47,
                 cutoff = 0):
    
    '''
    
    Function to determine the flux of chi 2 at Earth assuming that
    self-scattering is negligible, and ignoring decays
    
        Note that if there is no self-scattering and no decays then
        there will be no chi 1
    
    Parameters
    ----------
    mDM : Dark matter ground state mass [GeV]
    delta : Mass splitting [GeV]
    gp : Proton coupling
    gChi : Dark Matter Coupling
    mZ: Mediator Mass [GeV]
    mu : cos(angle) of jet relative to line-of-sight. The default is 1.
    dist: distance from black hole to Earth [cm]. Default is 1835.4 Mpc.
    MBH : Mass of Black Hole [Solar Mass]. The default is 1e8.
    Gamma_B :  Boost Factor of Blob. The default is 20.
    alpha_p : Power Law behavior in Blob. The default is 2.
    cp : Luminosity normalizing constant [s^{-1} sr^{-1}]. The default is 2.7e47.

    Returns
    -------
    E2vals : Energies of chi_2 particles [GeV]
    dphidE2vals : Flux of chi_2 particles [GeV^{-1} cm^{-2} s^{-1}]
    '''
    mp = 0.94  # GeV
    sigmaSM0, sigma10, sigma20 = sigma0valsFiniteMass(gp,gChi,mZ,mp,mDM,delta)
    Epthres = (delta**2 + 2*mp*mDM + 2*mDM*delta + 2*mp*delta)/(2*mDM)
    
    ESMedges = np.logspace(np.log10(Epthres), 7, 600)  # GeV
    E2edges = np.logspace(np.log10((mDM + delta)), 6, 702)  # GeV
    ESMvals = np.sqrt(ESMedges[:-1]*ESMedges[1:])
    E2vals = np.sqrt(E2edges[:-1]*E2edges[1:])
    
    dESM = ESMedges[1:] - ESMedges[:-1]
    dE2 = E2edges[1:] - E2edges[:-1]
    
    tot_int_num_dens = CF.BlackHoleIntDensity(MBH,mDM) #cm^{-2}
    
    ESMvals = np.transpose([ESMvals])
    dESM = np.transpose([dESM])
    
    TDMvals = E2vals - (mDM+delta)
    TSMvals = ESMvals - mp
    
    muVals = (-delta**2 + 2*mp*(delta+TDMvals) + 2*mDM*TDMvals + 2*delta*TSMvals + 2*TSMvals*TDMvals) \
        /(2 * sqrt(TSMvals*(2*mp+TSMvals)) * sqrt(TDMvals*(2*delta + 2*mDM + TDMvals)))
        
    muVals = muVals * np.heaviside(1 - muVals,1) * np.heaviside(1 + muVals,1)
        
    #print("muVals",muVals)
    
    dNSMdESM = CF.AGNProtonRate(Gamma_B,alpha_p,cp,muVals,ESMvals,cutoff) #GeV^{-1} s^{-1} sr^{-1}
    
    dsigmadTDM = dsigmadE2SMFiniteMass(sigmaSM0,mZ, ESMvals, E2vals, mp, mDM, delta)
    
    dphidE2vals = tot_int_num_dens/(dist**2) * np.sum(dESM*dNSMdESM*dsigmadTDM,axis=0)
    
    return(E2vals,dphidE2vals)

##========================================================
#Functions Considering a 2-body decay
##========================================================

def Momentum_Rest(mDM,delta,mZ):
    '''
    Function to determine the momentum of the decay products
        in the rest frame of chi 2

    Parameters
    ----------
    mDM : Ground State DM mass [GeV]
    delta : Mass Splitting [GeV]
    mZ : Mediator Mass [GeV]

    Returns
    -------
    p_r : Momentum in the chi_2 rest frame
    '''
    num = sqrt( (delta**2 - mZ**2) * ((2*mDM + delta)**2 - mZ**2))
    denom = 2 * (mDM + delta)
    p_r = num/denom
    return(p_r)


def Cos_Theta_Dec(mDM,delta,Echi1,Echi2,pr):
    '''
    Function to determine the angle of decay in the
        lab frame

    Parameters
    ----------
    mDM : Ground State DM mass [GeV]
    delta : Mass Splitting [GeV]
    Echi1 : Decay Chi 1 Energy [GeV]
    Echi2 : Original Chi 2 energy [GeV]
    pr: Momentum of Chi 1 in Chi 2 rest frame [GeV]

    Returns
    -------
    cos_theta
    '''
    Boost = Echi2/(mDM + delta)
    Beta = sqrt(1 - 1/Boost**2)
    
    num = Echi1 - (1/Boost) * sqrt(mDM**2 + pr**2)
    denom  = Beta * sqrt(Echi1**2 - mDM**2)
    
    cos_theta = num/denom
    return(cos_theta)

def Cos_Theta_Scat(mDM,delta,TSM,Tchi2,mSM = 0.94):
    '''
    Function to calculate the scattering angle in SM-DM
        interactions

    Parameters
    ----------
    mDM : Ground State DM mass [GeV]
    delta : Mass Splitting [GeV]
    TSM : Initial kinetic energy of SM particle [GeV]
    Tchi2 : Kinetic Energy of upscattered DM [GeV]
    mSM : SM mass [GeV]. The default is a proton.

    Returns
    -------
    cos_theta
    '''
    num = -delta**2 + 2*(mSM+TSM)*(delta+Tchi2) + 2*mDM*Tchi2
    denom = 2*sqrt(TSM * (2 * mSM + TSM)) * sqrt(Tchi2 * (2*delta + 2*mDM +Tchi2))
    
    cos_theta = num/denom
    return(cos_theta)
    


def SingleScatterFluxFiniteMassWDec(mDM,delta,gp,gChi,mZ, 
                 dist = 1835.4 * (3.86e24), MBH = 1e8, Gamma_B = 20, alpha_p = 2, cp = 2.7e47,
                 cutoff = 0):
    
    '''
    
    Function to determine the flux of chi 1 at Earth assuming that
    self-scattering is negligible, and including \chi_2 -> \chi_1 + Z' decays
    
        Note that if there is no self-scattering and everything decays
        there is only chi1
    
    Parameters
    ----------
    mDM : Dark matter ground state mass [GeV]
    delta : Mass splitting [GeV]
    gp : Proton coupling
    gChi : Dark Matter Coupling
    mZ: Mediator Mass [GeV]
    mu : cos(angle) of jet relative to line-of-sight. The default is 1.
    dist: distance from black hole to Earth [cm]. Default is 1835.4 Mpc.
    MBH : Mass of Black Hole [Solar Mass]. The default is 1e8.
    Gamma_B :  Boost Factor of Blob. The default is 20.
    alpha_p : Power Law behavior in Blob. The default is 2.
    cp : Luminosity normalizing constant [s^{-1} sr^{-1}]. The default is 2.7e47.

    Returns
    -------
    E2vals : Energies of chi_2 particles [GeV]
    dphidE2vals : Flux of chi_2 particles [GeV^{-1} cm^{-2} s^{-1}]
    '''
    
    mp = 0.94  # GeV
    sigmaSM0, sigma10, sigma20 = sigma0valsFiniteMass(gp,gChi,mZ,mp,mDM,delta)
    Epthres = (delta**2 + 2*mp*mDM + 2*mDM*delta + 2*mp*delta)/(2*mDM)
    
    ESMedges = np.logspace(np.log10(Epthres), 7, 600)  # GeV
    E1edges = np.logspace(np.log10(mDM),5,120)
    E2edges = np.logspace(np.log10((mDM + delta)), 6, 602)  # GeV
    ESMvals = np.sqrt(ESMedges[:-1]*ESMedges[1:])
    E1vals = np.sqrt(E1edges[:-1]*E1edges[1:])
    E2vals = np.sqrt(E2edges[:-1]*E2edges[1:])
    
    dPhidE1vals = np.array([])
    
    dESM = ESMedges[1:] - ESMedges[:-1]
    dE1 = E1edges[1:] - E1edges[:-1]
    dE2 = E2edges[1:] - E2edges[:-1]
    
    tot_int_num_dens = CF.BlackHoleIntDensity(MBH,mDM) #cm^{-2}
    
    ESMvals = np.transpose([ESMvals])
    dESM = np.transpose([dESM])
    
    TDM2vals = E2vals - (mDM+delta)
    TSMvals = ESMvals - mp
    
    if mZ > delta:
        pr = Momentum_Rest(mDM,delta,0)
    else:
        pr = Momentum_Rest(mDM,delta,mZ)
    BoostChi2 = E2vals/(mDM + delta) #Boost factors for chi2
    BetaChi2 = sqrt(1-1/BoostChi2**2)
        
    phi_scat_vals = np.linspace(0,2*pi,20)
    dphi = phi_scat_vals[1] - phi_scat_vals[0]
    
    
    prefactor = tot_int_num_dens / (4*pi*dist**2)
    for E1 in E1vals:
        #print("E1",E1)
        kin_req_1 = np.heaviside(E1 +BoostChi2*(BetaChi2 * pr - sqrt(mDM**2 +pr**2)),0) \
            *np.heaviside(BoostChi2*(BetaChi2 * pr + sqrt(mDM**2 +pr**2)) - E1,1)
            
        #print('kin 1',np.sum(kin_req_1))
            
        cos_theta_scat = Cos_Theta_Scat(mDM,delta,TSMvals,TDM2vals)
        cos_theta_dec = Cos_Theta_Dec(mDM,delta,E1,E2vals,pr)

        kin_req_2 = np.heaviside(cos_theta_scat + 1,0) * np.heaviside(1-cos_theta_scat,0)\
            *np.heaviside(cos_theta_dec + 1,0) * np.heaviside(1-cos_theta_dec,0)
        #print('kin 2', np.sum(kin_req_2))
        sin_theta_scat = np.sqrt((1 - cos_theta_scat**2)*kin_req_2)
        sin_theta_dec = sqrt((1 - cos_theta_dec**2)*kin_req_2)
        
        dsigmadTchi2 = dsigmadE2SMFiniteMass(sigmaSM0,mZ,ESMvals,
                                             E2vals,mp,mDM,delta)
        
        integrated_prod_rate = np.zeros((len(ESMvals),len(E2vals)))
        for phi_scat  in phi_scat_vals:
            muVals = (cos_theta_scat*cos_theta_dec + sin_theta_scat*sin_theta_dec *cos(phi_scat))*kin_req_2 
            integrated_prod_rate += CF.AGNProtonRate(Gamma_B,alpha_p,cp,muVals,ESMvals,cutoff) * dphi * kin_req_2
        
        #integrated_prod_rate = np.transpose([integrated_prod_rate])
        
        Integrand = integrated_prod_rate * dsigmadTchi2 * 1/(BoostChi2*BetaChi2*pr) * kin_req_1
        
        dPhidE1vals = np.append(dPhidE1vals,np.sum(prefactor*Integrand*dE2*dESM))
        
    return(E1vals, dPhidE1vals)


def dsigmaChi1dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E1,TSM,mSM = 0.94,Lambda = 0.77):
    '''
    Function for differential cross section of
        chi_1 - SM scattering with an SM particle at rest

    Parameters
    ----------
    gchi : Dark Matter Coupling / boson mass
    gSM : Standard Model Coupling / boson mass
    mZ : Mass of vector boson [GeV]
    mDM : Mass of Ground State DM [GeV]
    delta : Mass Splitting [GeV]
    E1 : Incoming energy of chi_1 [GeV]
    TSM : Outgoing kinetic energy of SM particle [GeV]

    Returns
    -------
    dsdTSM : Differential cross section [cm^2 GeV^{-1}]

    '''
    prefactor = gSM**2 * gchi**2 /(8 * pi * (E1**2 - mDM**2) * (mZ**2 + 2*mSM*TSM)**2) * (mDM/mSM) #GeV^{-6}
    first_term = -4 * E1**2 * mSM + delta**2 * (2*E1 + mSM - TSM)\
        +4*E1*mSM*TSM + 4*delta*E1*mDM + 2*TSM*(mSM**2 - mSM*TSM + mDM**2)#GeV^{3}
    
    s = mDM**2 + mSM**2 + 2*E1*mSM
    
    kinematic_req_1 = np.heaviside(s - (mDM + delta + mSM)**2,0)
    
    tplus,tminus = CF.tplusminus(mSM,mDM,delta,s)
    TSMmin = -tminus/(2*mSM)
    TSMmax = -tplus/(2*mSM)
    
    kinematic_req_2 = np.heaviside(TSM - TSMmin,0) * np.heaviside(TSMmax-TSM,0)
    
    qsq = 2 * mSM * TSM
    #Lambda = 0.77 #GeV
    xsq = qsq/Lambda**2
    FormFactor = (1+xsq)**(-2)
    
    Inv_GeV_to_cm = 1.98e-14
    
    dsdTSM = -prefactor * first_term * FormFactor**2\
        * kinematic_req_1*kinematic_req_2 * Inv_GeV_to_cm**2 #cm^{2} GeV^{-1}
    
    return(dsdTSM)

def dsigmaChi2dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E2,TSM,mSM = 0.94,Lambda = 0.77):
    '''
    Function for differential cross section of
        chi_2 - SM scattering with an SM particle at rest

    Parameters
    ----------
    gchi_MZ : Dark Matter Coupling
    gSM_MZ : Standard Model Coupling
    mZ : Mass of vector mediator [GeV]
    mDM : Mass of Ground State DM [GeV]
    delta : Mass Splitting [GeV]
    E2 : Incoming energy of chi_2 [GeV]
    TSM : Outgoing kinetic energy of SM particle [GeV]

    Returns
    -------
    dsdTSM : Differential cross section [cm^2 GeV^{-1}]

    '''
    s = (mDM + delta)**2 + mSM**2 + 2*E2*mSM
    tplus,tminus = CF.tplusminus(mSM,mDM,delta,s)
    TSMmin = -tminus/(2*mSM)
    TSMmax = -tplus/(2*mSM)
    kinematic_req = np.heaviside(TSM - TSMmin,0) * np.heaviside(TSMmax-TSM,0)
    
    qsq = 2 * mSM * TSM
    #Lambda = 0.77 #GeV
    xsq = qsq/Lambda**2
    FormFactor = (1+xsq)**(-2)
    
    prefactor = gSM**2 * gchi**2 /(8*pi*(E2**2 - (delta+mDM)**2) * (mZ**2 + 2*mSM*TSM)**2) * (mDM/mSM)
    first_term = -4*E2**2*mSM + 4*E2*mSM*TSM - 2*delta*E2*(delta + 2*mDM)\
        + 2*TSM*(mSM**2 - mSM*TSM + mDM**2) + delta**2 * (mSM + TSM)\
            + 4*delta*mDM*TSM
    
    Inv_GeV_to_cm = 1.98e-14
    
    dsdTSM = -prefactor * first_term *FormFactor**2\
        *kinematic_req *Inv_GeV_to_cm**2
        
    return(dsdTSM)

def dGammadEchi3Body(gSM,gChi,mf,mDM,delta,mZ,Echi1):
    '''
    Calculate the differential decay rate for
        chi2-> chi1 + f + Bar(f) in the rest
        frame of chi2.

    Parameters
    ----------
    gSM : Coupling of SM fermions to boson
    gChi : Coupling of DM to boson
    mf : mass of SM fermion (GeV)
    mDM : Mass of dark matter ground state (GeV)
    delta : Mass splitting (GeV)
    mZ : Mass of boson (GeV)
    Echi1 : Energy of chi_1

    Returns
    -------
    dGammadEchi1 = Differential decay rate (unitless)
    '''
    
    mffSq = (mDM + delta)**2 + mDM**2 - 2*(mDM+delta)*Echi1 #GeV
    '''
    print('min mee^2',np.min(mffSq))
    print('max mee^2', np.max(mffSq))
    print(np.sum(np.heaviside(-(mffSq-delta**2)*(mffSq - (delta + 2*mDM)**2),0)))
    
    wrong_vals = np.where((mffSq-delta**2)*(mffSq - (delta + 2*mDM)**2) < 0)
    
    try:
        print(mffSq[wrong_vals])
        print((mffSq[wrong_vals]-delta**2)*(mffSq[wrong_vals] - (delta + 2*mDM)**2))
    except:
        print("No Problem")
    '''
    
    num = gSM**2 * gChi**2 * sqrt(mffSq*(mffSq - 4*mf**2) * np.heaviside(mffSq - 4*mf**2,0)) * (2 * mf**2 + mffSq)\
        * (mffSq - delta**2) * sqrt((mffSq-delta**2)*(mffSq - (delta + 2*mDM)**2) * np.heaviside(delta**2 - mffSq,0)) \
            * (2 * mffSq + (delta + 2*mDM)**2) #GeV^{10}
    
    denom = 192 * pi**3 * mffSq**2 * (mffSq - mZ**2)**2 * (delta + mDM)**3 #GeV^{11}
    
    dGammadmffSq = (num/denom) #GeV^{-1}
    
    dGammadEchi1 = dGammadmffSq * 2 * (mDM + delta)
    
    return(dGammadEchi1)

def SingleScatterFluxFiniteMassWDec3Body(mDM,delta,gp,gChi,mZ,mf, 
                 dist = 1835.4 * (3.86e24), MBH = 1e8, Gamma_B = 20, alpha_p = 2, cp = 2.7e47,
                 cutoff = 0):
    
    '''
    
    Function to determine the flux of chi 1 at Earth assuming that
    self-scattering is negligible, including chi_2 -> chi_1 + f + \Bar{f} decays
    
        Note that if there is no self-scattering and everything decays
        there is only chi1
    
    Parameters
    ----------
    mDM : Dark matter ground state mass [GeV]
    delta : Mass splitting [GeV]
    gp : Proton coupling
    gChi : Dark Matter Coupling
    mZ: Mediator Mass [GeV]
    mf: Decay fermion mass [GeV]
    mu : cos(angle) of jet relative to line-of-sight. The default is 1.
    dist: distance from black hole to Earth [cm]. Default is 1835.4 Mpc.
    MBH : Mass of Black Hole [Solar Mass]. The default is 1e8.
    Gamma_B :  Boost Factor of Blob. The default is 20.
    alpha_p : Power Law behavior in Blob. The default is 2.
    cp : Luminosity normalizing constant [s^{-1} sr^{-1}]. The default is 2.7e47.

    Returns
    -------
    E2vals : Energies of chi_2 particles [GeV]
    dphidE2vals : Flux of chi_2 particles [GeV^{-1} cm^{-2} s^{-1}]
    '''
    
    mp = 0.94  # GeV
    sigmaSM0, sigma10, sigma20 = sigma0valsFiniteMass(gp,gChi,mZ,mp,mDM,delta)
    Epthres = (delta**2 + 2*mp*mDM + 2*mDM*delta + 2*mp*delta)/(2*mDM)
    
    E1Vals = np.transpose([np.logspace(np.log10(mDM)+0.1,5,40)])
    dPhi1dE1vals = np.zeros(len(E1Vals))
    
    tot_int_num_dens = CF.BlackHoleIntDensity(MBH,mDM) #cm^{-2}
    
    TSMmin = Epthres - mp #GeV
    TSMmax = 1e7 #GeV
    SMpower = 0.9
    
    E1rMin = mDM #GeV
    E1rMax = mDM + (delta**2 - 4 * mf**2)/(2*(mDM + delta)) #GeV
    
    Tchi2Min = 0.1 #GeV
    Tchi2Max = 1e6 #GeV
    Chi2power = 0.9
    
    num_trials = int(1e6)
    E1rTrialVals = np.linspace(E1rMin,E1rMax,num_trials)
    dE1rTrial = (E1rTrialVals[-1] - E1rTrialVals[0])/num_trials
    GammaDecTot = np.sum(dGammadEchi3Body(gp,gChi,mf,mDM,delta,mZ,E1rTrialVals)*dE1rTrial) #GeV
    print("GammaDecTot",GammaDecTot)
    #print("dE",dE1rTrial)
    #print(E1rTrialVals)
    
    prefactor = tot_int_num_dens/(4*pi*dist**2) #cm^{-4}
    
    
    num_repeats = 10
    
    for itteration in range(num_repeats):
        #print(itteration)
        
        num_points = int(1e5)
        
        TSMvals = (rand.random(num_points)* (TSMmax**(1-SMpower) - TSMmin**(1-SMpower))\
                   + TSMmin**(1-SMpower))**(1/(1-SMpower))#proton kinetic energy
        E1rvals = rand.random(num_points) * (E1rMax - E1rMin) + E1rMin #chi 1 energies in chi 2 rest frame
        Tchi2vals = (rand.random(num_points)* (Tchi2Max**(1-Chi2power) - Tchi2Min**(1-Chi2power))\
                   + Tchi2Min**(1-Chi2power))**(1/(1-Chi2power))#proton kinetic energy
        phivals = 2*pi*rand.random(num_points)
        
        TSMweight = 1/(1 - SMpower) * (TSMmax**(1-SMpower) - TSMmin**(1-SMpower)) * TSMvals**SMpower
        TChi2weight = 1/(1 - Chi2power) * (Tchi2Max**(1-Chi2power) - Tchi2Min**(1-Chi2power)) * Tchi2vals**Chi2power
        
        diff_element = TSMweight * (E1rMax - E1rMin)* TChi2weight * 2*pi/(num_points*num_repeats) #GeV^{3}
        
        
        ESMvals = TSMvals + mp
        Echi2vals = Tchi2vals + (mDM + delta)
        
        BoostChi2 = Echi2vals/(mDM + delta)
        BetaChi2 = sqrt(1 - 1/BoostChi2**2)
        
        prVals = sqrt(E1rvals**2 - mDM**2) #GeV
        cos_theta_scat = Cos_Theta_Scat(mDM,delta,TSMvals,Tchi2vals)
        
        
        kin_req_1 = np.heaviside(E1Vals +BoostChi2*(BetaChi2 * prVals - sqrt(mDM**2 +prVals**2)),0) \
            *np.heaviside(BoostChi2*(BetaChi2 * prVals + sqrt(mDM**2 +prVals**2)) - E1Vals,1)     
        
        cos_theta_dec = Cos_Theta_Dec(mDM,delta,E1Vals,Echi2vals,prVals)
        kin_req_2 = np.heaviside(cos_theta_scat + 1,0) * np.heaviside(1-cos_theta_scat,0)\
            *np.heaviside(cos_theta_dec + 1,0) * np.heaviside(1-cos_theta_dec,0)
            
        sin_theta_scat = np.sqrt((1 - cos_theta_scat**2)*kin_req_2)
        sin_theta_dec = sqrt((1 - cos_theta_dec**2)*kin_req_2)
        
        dsigmadTchi2 = dsigmadE2SMFiniteMass(sigmaSM0,mZ,ESMvals,
                                             Echi2vals,mp,mDM,delta) #cm^{2} GeV^{-1}
        
        muVals = (cos_theta_scat*cos_theta_dec + sin_theta_scat*sin_theta_dec *cos(phivals))*kin_req_2 
    
        DiffProtonRate = CF.AGNProtonRate(Gamma_B,alpha_p,cp,muVals,ESMvals,cutoff) #GeV^{-1} s^{-1}
        
        dGammadE1r = dGammadEchi3Body(gp,gChi,mf,mDM,delta,mZ,E1rvals) #unitless
        
        #print("min",np.min(BoostChi2*BetaChi2*prVals))
        #print("max",np.max(BoostChi2*BetaChi2*prVals))
        
        Integrand = DiffProtonRate * dsigmadTchi2 * (1/GammaDecTot) * dGammadE1r\
            * (1/(BoostChi2*BetaChi2*prVals + np.heaviside(-BoostChi2*BetaChi2*prVals,1))) * np.heaviside(BoostChi2*BetaChi2*prVals,0)\
            * kin_req_1 * kin_req_2 #cm^{2} * s^{-1} GeV^{-4}
        
        dPhi1dE1vals += np.sum(prefactor*Integrand*diff_element,axis = 1)
        
        del Integrand,diff_element,DiffProtonRate,dGammadE1r,kin_req_1,kin_req_2,cos_theta_dec

        #print(dPhi1dE1vals[0],dPhi1dE1vals[-1])
        
    
    return(E1Vals,dPhi1dE1vals)
    
    

#Make Plots of the Fluxes
'''
mDM = 0.1  # GeV
delta = 0.2
prod = 1e-3

mZ = mDM #GeV
mp = 0.94  # GeV
me = 511e-6
gp = 1
gChi = prod * mZ**2



fig2 = plt.figure("Chi2")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("$d \Phi_{2}/dE_{\chi_{2}}$ [$\mathrm{cm^{-2} s^{-1} GeV^{-1}}$]")
plt.xlabel("$E_{\chi_{2}}$ [GeV]")
plt.title("$m_{\chi}$ = "+str(mDM) + "GeV ; $\delta$ ="+str(delta) + "GeV ; $g_p g_{\chi}/M_{Z'}^2$ ="+str(prod)+"$GeV^{-2}$ ; Single Scatter")

E2valsS,dPhi2dE2S = SingleScatterFluxFiniteMassNoDec(mDM,delta,gp,gChi,mZ)
                                    #dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)


if delta > mZ:
    pr = Momentum_Rest(mDM,delta,mZ)
elif delta < mZ:
    pr = Momentum_Rest(mDM,delta,0)
BoostChi2 = E2valsS/(mDM + delta)
BetaChi2 = sqrt(1-1/BoostChi2**2)

E1CalcVals = np.transpose([np.logspace(np.log10(mDM),5,200)])
dE2 = E2valsS[1:]- E2valsS[:-1]
dE2 = np.append(dE2,dE2[-1])

kin_req = np.heaviside(E1CalcVals + BoostChi2*BetaChi2*pr - BoostChi2*sqrt(mDM**2 +pr**2),0)\
    * np.heaviside(-E1CalcVals + BoostChi2*BetaChi2*pr + BoostChi2*sqrt(mDM**2 + pr**2),0)
    
Phi1CalcVals = 1/2 * np.sum(dPhi2dE2S* 1/(BoostChi2*BetaChi2*pr) * kin_req*dE2,axis = 1)
                                    

plt.figure("Chi2")
plt.plot(E2valsS,dPhi2dE2S,label = "TXS, $m_{Z'}$ ="+str(mZ) + " GeV")
plt.legend()


fig2 = plt.figure("Chi1")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("$d \Phi_{1}/dE_{\chi_{1}}$ [$\mathrm{cm^{-2} s^{-1} GeV^{-1}}$]")
plt.xlabel("$E_{\chi_{1}}$ [GeV]")
plt.title("$m_{\chi}$ = "+str(mDM) + "GeV ; $\delta$ ="+str(delta) + "GeV ; $g_p g_{\chi}/M_{Z'}^2$ ="+str(prod)+"$GeV^{-2}$ ; Single Scatter")


if delta > mZ or delta < 2* me:
    E1valsS,dPhi1dE1S = SingleScatterFluxFiniteMassWDec(mDM,delta,gp,gChi,mZ)
                                                        #dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
elif delta < mZ and delta > 2*me:
    E1valsS,dPhi1dE1S = SingleScatterFluxFiniteMassWDec3Body(mDM,delta,gp,gChi,mZ,me)
                                        #dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
plt.figure("Chi1")
plt.plot(E1valsS,dPhi1dE1S,label = "TXS, $m_{Z'}$ ="+str(round(mZ,3)) + " GeV")
#plt.plot(E1CalcVals,Phi1CalcVals,label = "Calc")
plt.legend()

fig3 = plt.figure("Combined")

plt.xscale('log')
plt.yscale('log')
plt.ylabel("$d \Phi/dE_{\chi}$ [$\mathrm{cm^{-2} s^{-1} GeV^{-1}}$]")
plt.xlabel("$E_{\chi_{1}}$ [GeV]")
plt.title("TXS 056+0506 ; $m_{\chi}$ = "+str(round(mDM,2)) + "GeV ; $m_{Z^{\prime}}$ = "+str(round(mZ,3))
          +" GeV ; $g_p g_{\chi}$ =1e"+str(round(np.log10(gp*gChi),3)))

plt.plot(E1valsS,dPhi1dE1S,"--",color = "r",label = "$\delta$ ="+str(round(delta,3)) + "GeV ; $\chi_{1}$")
plt.plot(E2valsS,dPhi2dE2S,"--",color = "b",label = "$\delta$ ="+str(round(delta,3)) + "GeV ; $\chi_{2}$")
plt.legend(fontsize = 6,loc = "lower left")

plt.xlim([1e-1,1e3])
'''

#Get Exclusions for scattering and decays

Inv_GeV_to_cm = 1.98e-14
mp = 0.94
me = 511e-6
des_Events = 1000
des_Events_eHigh = 3
des_Events_eMid = 4.5
des_Events_eLow = 20

des_Events_eHigh_NoAng = 6
des_Events_eMid_NoAng = 51
des_Events_eLow_NoAng = 150

#mZ = 500 #GeV

guess_prod = 1 #Guess at g_p * g_chi
mDMlist = np.logspace(-3,0,12)
deltaVals = np.array([1])#np.logspace(-2,1,4)#np.logspace(-4,0,3)
delta_ratio = 0.4
mZ_ratio = 10

colors = ["r","g","b"]
fig = plt.figure("SK Exclusions Sing Scat Elec")
sources = ['TXS']#["TXS","NGC"]

#file = open("SK_Electron_Scattering_mZ_"+str(mZ)+"GeV_NoDIS.csv","w")



first_string = "TXS, " + (3*len(deltaVals)-1) * ", " + "NGC, " + 2*(len(deltaVals)-1) * ", " +"\n"
second_string = ""
third_string = ""
for delta in deltaVals:
    second_string += "delta ="+str(delta)+" GeV, , ,"
    third_string += "m_DM [GeV], g_{SM} g_{Chi} Chi 1 Scattering, g_{SM} g_{Chi} Chi 2 Scattering,"
    
second_string = second_string*2
third_string = third_string *2
    
second_string += "\n"
third_string += "\n"

#file.write(first_string)
#file.write(second_string)
#file.write(third_string)

CouplingExclude1 = np.array([])
CouplingExclude2 = np.array([])

for mDM in mDMlist:
    mZ = mDM*mZ_ratio
    
    
    for source in sources:
        
        for delta in deltaVals:
            delta = mDM*delta_ratio
    
        
            #color = colors[np.where(delta == deltaVals)[0][0]]
            
        
        
        
            print('mDM', mDM)
            print('delta',delta)
            mu_chi_p = (mDM*mp)/(mDM + mp)
            gchi = guess_prod
            gSM = 1
            
            #Get Flux properties    
            if source == "TXS":
                E2vals,dPhi2dE2 = SingleScatterFluxFiniteMassNoDec(mDM,delta,gSM,gchi,mZ)
                if delta > mZ:
                    E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec(mDM,delta,gSM,gchi,mZ)
                elif delta < mZ and delta > 2*me:
                    E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec3Body(mDM,delta,gSM,gchi,mZ,me)
                    E1vals = E1vals[:,0]
                elif mZ > delta and delta < 2*me:
                    E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec(mDM,delta,gSM,gchi,mZ)
                else:
                    print("No Decays")
                    E1vals = np.logspace(np.log10(mDM)+0.5,5,100)
                    dPhi1dE1 = np.zeros(len(E1vals))
                                                          
            if source == "NGC":
                E2vals,dPhi2dE2 = SingleScatterFluxFiniteMassNoDec(mDM,delta,gSM,gchi,mZ,
                                                           dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
                                                           #cutoff = 1e5)
                if delta > mZ:
                    E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec(mDM,delta,gSM,gchi,mZ,
                                                           dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
                                                           #cutoff = 1e5)
                elif delta < mZ and delta>2*me:
                    E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec3Body(mDM,delta,gSM,gchi,mZ,me,
                                                           dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
                                                            #cutoff = 1e5
                    E1vals = E1vals[:,0]
                
                elif delta < mZ and delta < 2*me:
                    E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec(mDM,delta,gSM,gchi,mZ,
                                                           dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
                else:
                    print("No Decays")
                    E1vals = np.logspace(np.log10(mDM)+0.5,5,100)
                    dPhi1dE1 = np.zeros(len(E1vals))
            
            
            dE2 = E2vals[1:]-E2vals[:-1]
            dE2 = np.append(dE2,dE2[-1])
            
            
            
            num_protons = 22.5*1e9 * 6e23/2
            
            
            #Get Detector Scattering Properties for Electrons
            TSMvalsHigh = np.transpose([np.linspace(20,1000,10000)]) #GeV
            TSMvalsMid = np.transpose([np.linspace(1.33,20,10000)]) #GeV
            TSMvalsLow = np.transpose([np.linspace(0.1,1.33,10000)]) #GeV
            
            ThetaMaxHigh = 5 * pi/180
            ThetaMaxMid =  7* pi/180
            ThetaMaxLow = 24 * pi/180
            
            dsigma2dTSMHigh = dsigmaChi2dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E2vals,TSMvalsHigh,mSM = 9.1e-4,Lambda = 1e10)
            dsigma2dTSMMid = dsigmaChi2dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E2vals,TSMvalsMid,mSM = 9.1e-4,Lambda = 1e10)
            dsigma2dTSMLow = dsigmaChi2dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E2vals,TSMvalsLow,mSM = 9.1e-4,Lambda = 1e10)
            
            #Note, we ignore the mass of the electron for angular calculations
            
            cosTheta2High = (2*TSMvalsHigh*(E2vals) - 2*mDM*delta - delta**2)\
                /(2*TSMvalsHigh * sqrt(E2vals**2 - (mDM+delta)**2))
            cosTheta2Mid = (2*TSMvalsMid*(E2vals) - 2*mDM*delta - delta**2)\
                /(2*TSMvalsMid * sqrt(E2vals**2 - (mDM+delta)**2))
            cosTheta2Low = (2*TSMvalsLow*(E2vals) - 2*mDM*delta - delta**2)\
                /(2*TSMvalsLow * sqrt(E2vals**2 - (mDM+delta)**2))
            
            AngleReq2High = np.heaviside(cosTheta2High - cos(ThetaMaxHigh),1)
            AngleReq2Mid = np.heaviside(cosTheta2Mid - cos(ThetaMaxMid),1)
            AngleReq2Low = np.heaviside(cosTheta2Low - cos(ThetaMaxLow),1)
            
            dratedTSMHigh2 = num_protons * np.sum(dPhi2dE2*dsigma2dTSMHigh*dE2
                                                *AngleReq2High, axis = 1)
            dratedTSMMid2 = num_protons * np.sum(dPhi2dE2*dsigma2dTSMMid*dE2
                                                *AngleReq2Mid, axis = 1)
            dratedTSMLow2 = num_protons * np.sum(dPhi2dE2*dsigma2dTSMLow*dE2
                                                *AngleReq2Low, axis = 1)
            
            dratedTSMHigh2NoAng = num_protons * np.sum(dPhi2dE2*dsigma2dTSMHigh*dE2, axis = 1)
            dratedTSMMid2NoAng = num_protons * np.sum(dPhi2dE2*dsigma2dTSMMid*dE2, axis = 1)
            dratedTSMLow2NoAng = num_protons * np.sum(dPhi2dE2*dsigma2dTSMLow*dE2, axis = 1)
    
            dTSMHigh = TSMvalsHigh[1] - TSMvalsHigh[0]
            dTSMMid = TSMvalsMid[1] - TSMvalsMid[0]
            dTSMLow = TSMvalsLow[1] - TSMvalsLow[0]

            
            ElectronCountsHigh2 = np.sum(dratedTSMHigh2*dTSMHigh)*2500*86400
            ElectronCountsMid2 = np.sum(dratedTSMMid2*dTSMMid)*2500*86400
            ElectronCountsLow2 = np.sum(dratedTSMLow2*dTSMLow)*2500*86400
            
            ElectronCountsHigh2NoAng = np.sum(dratedTSMHigh2NoAng*dTSMHigh)*2500*86400
            ElectronCountsMid2NoAng = np.sum(dratedTSMMid2NoAng*dTSMMid)*2500*86400
            ElectronCountsLow2NoAng = np.sum(dratedTSMLow2NoAng*dTSMLow)*2500*86400
            
            exclude_prod_High2 = guess_prod * (des_Events_eHigh/ElectronCountsHigh2)**(1/4)
            exclude_prod_Mid2 = guess_prod * (des_Events_eMid/ElectronCountsMid2)**(1/4)
            exclude_prod_Low2 = guess_prod * (des_Events_eLow/ElectronCountsLow2)**(1/4)
            
            exclude_prod_High2NoAng = guess_prod * (des_Events_eHigh_NoAng/ElectronCountsHigh2NoAng)**(1/4)
            exclude_prod_Mid2NoAng = guess_prod * (des_Events_eMid_NoAng/ElectronCountsMid2NoAng)**(1/4)
            exclude_prod_Low2NoAng = guess_prod * (des_Events_eLow/ElectronCountsLow2NoAng)**(1/4)
            
            if (np.min([exclude_prod_High2,exclude_prod_Mid2,exclude_prod_Low2]) 
                >np.min([exclude_prod_High2NoAng,exclude_prod_Mid2NoAng,exclude_prod_Low2NoAng])):
                print("Chi 2: Angular Cut Worse")
            
            coup2 = np.min([exclude_prod_High2,exclude_prod_Mid2,exclude_prod_Low2,
                            exclude_prod_High2NoAng,exclude_prod_Mid2NoAng,exclude_prod_Low2NoAng])
            if coup2 == np.inf:
                coup2 = 1e10
            
            CouplingExclude2 = np.append(CouplingExclude2,coup2)
            
            
            dE1 = E1vals[1:]-E1vals[:-1]
            dE1 = np.append(dE1,dE1[-1])
            
            dsigma1dTSMHigh = dsigmaChi1dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E1vals,TSMvalsHigh,mSM = 9.1e-4,Lambda = 1e10)
            dsigma1dTSMMid = dsigmaChi1dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E1vals,TSMvalsMid,mSM = 9.1e-4,Lambda = 1e10)
            dsigma1dTSMLow = dsigmaChi1dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E1vals,TSMvalsLow,mSM = 9.1e-4,Lambda = 1e10)
            
            cosTheta1High = (delta**2 + 2*TSMvalsHigh*E1vals + 2*delta*mDM)\
                /(2*TSMvalsHigh * sqrt(E1vals**2 - mDM**2))
            cosTheta1Mid = (delta**2 + 2*TSMvalsMid*E1vals + 2*delta*mDM)\
                /(2*TSMvalsMid * sqrt(E1vals**2 - mDM**2))
            cosTheta1Low = (delta**2 + 2*TSMvalsLow*E1vals + 2*delta*mDM)\
                /(2*TSMvalsLow * sqrt(E1vals**2 - mDM**2))
                
            AngleReq1High = np.heaviside(cosTheta1High - cos(ThetaMaxHigh),1)
            AngleReq1Mid = np.heaviside(cosTheta1Mid - cos(ThetaMaxMid),1)
            AngleReq1Low = np.heaviside(cosTheta1Low - cos(ThetaMaxLow),1)
            
            dratedTSMHigh1 = num_protons * np.sum(dPhi1dE1*dsigma1dTSMHigh*dE1
                                                *AngleReq1High, axis = 1)
            dratedTSMMid1 = num_protons * np.sum(dPhi1dE1*dsigma1dTSMMid*dE1
                                                *AngleReq1Mid, axis = 1)
            dratedTSMLow1 = num_protons * np.sum(dPhi1dE1*dsigma1dTSMLow*dE1
                                                *AngleReq1Low, axis = 1)
            
            dratedTSMHigh1NoAng = num_protons * np.sum(dPhi1dE1*dsigma1dTSMHigh*dE1, axis = 1)
            dratedTSMMid1NoAng = num_protons * np.sum(dPhi1dE1*dsigma1dTSMMid*dE1, axis = 1)
            dratedTSMLow1NoAng = num_protons * np.sum(dPhi1dE1*dsigma1dTSMLow*dE1, axis = 1)
            
            ElectronCountsHigh1 = np.sum(dratedTSMHigh1*dTSMHigh)*2500*86400
            ElectronCountsMid1 = np.sum(dratedTSMMid1*dTSMMid)*2500*86400
            ElectronCountsLow1 = np.sum(dratedTSMLow1*dTSMLow)*2500*86400
            
            ElectronCountsHigh1NoAng = np.sum(dratedTSMHigh1NoAng*dTSMHigh)*2500*86400
            ElectronCountsMid1NoAng = np.sum(dratedTSMMid1NoAng*dTSMMid)*2500*86400
            ElectronCountsLow1NoAng = np.sum(dratedTSMLow1NoAng*dTSMLow)*2500*86400
            
            exclude_prod_High1 = guess_prod * (des_Events_eHigh/ElectronCountsHigh1)**(1/4)
            exclude_prod_Mid1 = guess_prod * (des_Events_eMid/ElectronCountsMid1)**(1/4)
            exclude_prod_Low1 = guess_prod * (des_Events_eLow/ElectronCountsLow1)**(1/4)
            
            exclude_prod_High1NoAng = guess_prod * (des_Events_eHigh_NoAng/ElectronCountsHigh1NoAng)**(1/4)
            exclude_prod_Mid1NoAng = guess_prod * (des_Events_eMid_NoAng/ElectronCountsMid1NoAng)**(1/4)
            exclude_prod_Low1NoAng = guess_prod * (des_Events_eLow/ElectronCountsLow1NoAng)**(1/4)
            
            if (np.min([exclude_prod_High1,exclude_prod_Mid1,exclude_prod_Low1]) 
                >np.min([exclude_prod_High1NoAng,exclude_prod_Mid1NoAng,exclude_prod_Low1NoAng])):
                print("Chi 1: Angular Cut Worse")
            
            coup1 = np.min([exclude_prod_High1,exclude_prod_Mid1,exclude_prod_Low1,
                            exclude_prod_High1NoAng,exclude_prod_Mid1NoAng,exclude_prod_Low1NoAng])
            if coup1 == np.inf:
                coup1 = 1e10
            print("coup 1", coup1, "coup 2",coup2)
            CouplingExclude1 = np.append(CouplingExclude1,coup1)
            
            #file.write(str(mDM) +","+ str(coup1)+"," + str(coup2) +",")
            
        
        
        
    #file.write('\n')
    
#file.close()



file = open("yVals-Exisiting-Bounds.csv","r")
mDMexisting = np.array([])
yExisting = np.array([])
line_index = 0
for line in file:
    line_index += 1
    if line_index >1:
        line = line.split(',')
        mDMexisting = np.append(mDMexisting,float(line[0]))
        yExisting = np.append(yExisting,float(line[1]))
file.close()

file = open("y-vals-thermal-relic.csv","r")
mDMthermal = np.array([])
ythermal = np.array([])
line_index = 0
for line in file:
    line_index += 1
    if line_index > 1:
        line = line.split(',')
        mDMthermal = np.append(mDMthermal,float(line[0]))
        ythermal = np.append(ythermal,float(line[1]))
file.close()



fig = plt.figure("yVals")

#plt.fill_between(mDMexisting,yExisting,100,label = "Existing Bounds",color = "grey",alpha = 0.5)
#plt.plot(mDMthermal,ythermal,label = "Thermal inelastic dark matter",color = "purple")

yvals1 = CouplingExclude1**2/(4*pi*0.3**2) * (1/mZ_ratio)**4
yvals2 = CouplingExclude2**2/(4*pi*0.3**2) * (1/mZ_ratio)**4

plt.plot(mDMlist,yvals1, label = "$\chi_{1}$ "+source)
#plt.plot(mDMlist,yvals2, label = "$\chi_{2}$ (Ignoring Decays)")
plt.xscale('log')
plt.yscale('log')

plt.xlabel('$m_{\chi}$[GeV]')
plt.ylabel("y = $\epsilon^2 \\alpha_{D} (m_{\chi}/m_{Z'})^4$")
plt.legend()

plt.title('$\delta$ = '+str(round(delta_ratio,1))+"$m_{DM}$ ; $m_{Z'}$ ="+str(mZ_ratio)+"$m_{DM}$")

'''
file = open("EpsilonVals-Existing-Bounds.csv","r")
mDMexisting = np.array([])
epsilonExisting = np.array([])
line_index = 0
for line in file:
    line_index += 1
    if line_index > 1:
        line = line.split(',')
        mDMexisting = np.append(mDMexisting,float(line[0]))
        epsilonExisting = np.append(epsilonExisting,float(line[1]))
file.close()

file = open("eps-vals-thermal-relic.csv","r")
mDMthermal1 = np.array([])
epsthermal1 = np.array([])
mDMthermal2 = np.array([])
epsthermal2 = np.array([])
line_index = 0

first_bound = True
for line in file:
    line_index += 1
    if line_index > 1:
        line = line.split(',')
        if line_index == 2:
            mDMthermal1 = np.append(mDMthermal1,float(line[0]))
            epsthermal1 = np.append(epsthermal1,float(line[1]))
        else:
            if first_bound:
                if float(line[0]) < mDMthermal1[-1]:
                    first_bound = False
            
            if first_bound:
                mDMthermal1 = np.append(mDMthermal1,float(line[0]))
                epsthermal1 = np.append(epsthermal1,float(line[1]))
            else:
                mDMthermal2 = np.append(mDMthermal2,float(line[0]))
                epsthermal2 = np.append(epsthermal2,float(line[1]))
file.close()

epsthermal2interp = np.interp(mDMthermal1,mDMthermal2,epsthermal2)            

fig = plt.figure("epsVals")
plt.fill_between(mDMexisting,epsilonExisting,1000,label = "Exisiting Bounds", color ="grey", alpha = 0.5)
#plt.plot(mDMthermal1,epsthermal1,label = "Thermal Relic")
#plt.plot(mDMthermal2, epsthermal2, label = "Thermal Relic")
plt.fill_between(mDMthermal1,epsthermal1,epsthermal2interp,label = "Thermal Relic", color = "purple", alpha = 0.5)



epsVals = CouplingExclude1/(0.3 * sqrt(4*pi*0.5))
plt.plot(mDMlist,epsVals, label = "$\chi_{1}$ "+source)
plt.xscale('log')
plt.yscale('log')

plt.xlabel('$m_{\chi}$[GeV]')
plt.ylabel("$\epsilon$")
plt.legend()

plt.title('$\delta$ = '+str(round(delta_ratio,1))+"$m_{DM}$ ; $m_{Z'}$ ="+str(mZ_ratio)
          +"$m_{DM}$ ; $\\alpha_{D}$ = 0.5")

'''


'''
#Make Contour Plot for Exclusions
Inv_GeV_to_cm = 1.98e-14
mp = 0.94
me = 511e-6
des_Events = 1000
des_Events_eHigh = 3
des_Events_eMid = 4.5
des_Events_eLow = 20

des_Events_eHigh_NoAng = 6
des_Events_eMid_NoAng = 51
des_Events_eLow_NoAng = 150

#mZ = 500 #GeV

guess_prod = 1 #Guess at g_p * g_chi
mDMlist = np.logspace(-6,1,15)
deltaVals = np.logspace(-6,1,14)#np.logspace(-4,0,3)

mZ = 1e-2 #GeV

colors = ["r","g","b"]
fig = plt.figure("SK Exclusions Sing Scat Elec")
source = "TXS"#["TXS","NGC"]

#file = open("SK_Electron_Scattering_mZ_"+str(mZ)+"GeV_NoDIS.csv","w")



first_string = "TXS, " + (3*len(deltaVals)-1) * ", " + "NGC, " + 2*(len(deltaVals)-1) * ", " +"\n"
second_string = ""
third_string = ""
for delta in deltaVals:
    second_string += "delta ="+str(delta)+" GeV, , ,"
    third_string += "m_DM [GeV], g_{SM} g_{Chi} Chi 1 Scattering, g_{SM} g_{Chi} Chi 2 Scattering,"
    
second_string = second_string*2
third_string = third_string *2
    
second_string += "\n"
third_string += "\n"

#file.write(first_string)
#file.write(second_string)
#file.write(third_string)

CouplingExclude1 = np.zeros((len(mDMlist),len(deltaVals)))
CouplingExclude2 = np.zeros((len(mDMlist),len(deltaVals)))


mDM_index = -1

for mDM in mDMlist:
    mDM_index += 1
    delta_index = -1
    
    
    for delta in deltaVals:
        delta_index += 1

        print('mDM', mDM)
        print('delta',delta)
        mu_chi_p = (mDM*mp)/(mDM + mp)
        gchi = guess_prod
        gSM = 1
        
        #Get Flux properties    
        if source == "TXS":
            E2vals,dPhi2dE2 = SingleScatterFluxFiniteMassNoDec(mDM,delta,gSM,gchi,mZ)
            if delta > mZ:
                E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec(mDM,delta,gSM,gchi,mZ)
            elif delta < mZ and delta > 2*me:
                E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec3Body(mDM,delta,gSM,gchi,mZ,me)
                E1vals = E1vals[:,0]
            elif mZ > delta and delta < 2*me:
                E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec(mDM,delta,gSM,gchi,mZ)
            else:
                print("No Decays")
                E1vals = np.logspace(np.log10(mDM)+0.5,5,100)
                dPhi1dE1 = np.zeros(len(E1vals))
                                                      
        if source == "NGC":
            E2vals,dPhi2dE2 = SingleScatterFluxFiniteMassNoDec(mDM,delta,gSM,gchi,mZ,
                                                       dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
                                                       #cutoff = 1e5)
            if delta > mZ:
                E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec(mDM,delta,gSM,gchi,mZ,
                                                       dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
                                                       #cutoff = 1e5)
            elif delta < mZ and delta>2*me:
                E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec3Body(mDM,delta,gSM,gchi,mZ,me,
                                                       dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
                                                        #cutoff = 1e5
                E1vals = E1vals[:,0]
            
            elif delta < mZ and delta < 2*me:
                E1vals,dPhi1dE1 = SingleScatterFluxFiniteMassWDec(mDM,delta,gSM,gchi,mZ,
                                                       dist = 14 * (3.86e24), MBH = 1e7, Gamma_B = 1, alpha_p = 2, cp = 1e46)
            else:
                print("No Decays")
                E1vals = np.logspace(np.log10(mDM)+0.5,5,100)
                dPhi1dE1 = np.zeros(len(E1vals))
        
        
        dE2 = E2vals[1:]-E2vals[:-1]
        dE2 = np.append(dE2,dE2[-1])
        
        
        
        num_protons = 22.5*1e9 * 6e23/2
        
        
        #Get Detector Scattering Properties for Electrons
        TSMvalsHigh = np.transpose([np.linspace(20,1000,10000)]) #GeV
        TSMvalsMid = np.transpose([np.linspace(1.33,20,10000)]) #GeV
        TSMvalsLow = np.transpose([np.linspace(0.1,1.33,10000)]) #GeV
        
        ThetaMaxHigh = 5 * pi/180
        ThetaMaxMid =  7* pi/180
        ThetaMaxLow = 24 * pi/180
        
        dsigma2dTSMHigh = dsigmaChi2dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E2vals,TSMvalsHigh,mSM = 9.1e-4,Lambda = 1e10)
        dsigma2dTSMMid = dsigmaChi2dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E2vals,TSMvalsMid,mSM = 9.1e-4,Lambda = 1e10)
        dsigma2dTSMLow = dsigmaChi2dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E2vals,TSMvalsLow,mSM = 9.1e-4,Lambda = 1e10)
        
        #Note, we ignore the mass of the electron for angular calculations
        
        cosTheta2High = (2*TSMvalsHigh*(E2vals) - 2*mDM*delta - delta**2)\
            /(2*TSMvalsHigh * sqrt(E2vals**2 - (mDM+delta)**2))
        cosTheta2Mid = (2*TSMvalsMid*(E2vals) - 2*mDM*delta - delta**2)\
            /(2*TSMvalsMid * sqrt(E2vals**2 - (mDM+delta)**2))
        cosTheta2Low = (2*TSMvalsLow*(E2vals) - 2*mDM*delta - delta**2)\
            /(2*TSMvalsLow * sqrt(E2vals**2 - (mDM+delta)**2))
        
        AngleReq2High = np.heaviside(cosTheta2High - cos(ThetaMaxHigh),1)
        AngleReq2Mid = np.heaviside(cosTheta2Mid - cos(ThetaMaxMid),1)
        AngleReq2Low = np.heaviside(cosTheta2Low - cos(ThetaMaxLow),1)
        
        dratedTSMHigh2 = num_protons * np.sum(dPhi2dE2*dsigma2dTSMHigh*dE2
                                            *AngleReq2High, axis = 1)
        dratedTSMMid2 = num_protons * np.sum(dPhi2dE2*dsigma2dTSMMid*dE2
                                            *AngleReq2Mid, axis = 1)
        dratedTSMLow2 = num_protons * np.sum(dPhi2dE2*dsigma2dTSMLow*dE2
                                            *AngleReq2Low, axis = 1)
        
        dratedTSMHigh2NoAng = num_protons * np.sum(dPhi2dE2*dsigma2dTSMHigh*dE2, axis = 1)
        dratedTSMMid2NoAng = num_protons * np.sum(dPhi2dE2*dsigma2dTSMMid*dE2, axis = 1)
        dratedTSMLow2NoAng = num_protons * np.sum(dPhi2dE2*dsigma2dTSMLow*dE2, axis = 1)

        dTSMHigh = TSMvalsHigh[1] - TSMvalsHigh[0]
        dTSMMid = TSMvalsMid[1] - TSMvalsMid[0]
        dTSMLow = TSMvalsLow[1] - TSMvalsLow[0]

        
        ElectronCountsHigh2 = np.sum(dratedTSMHigh2*dTSMHigh)*2500*86400
        ElectronCountsMid2 = np.sum(dratedTSMMid2*dTSMMid)*2500*86400
        ElectronCountsLow2 = np.sum(dratedTSMLow2*dTSMLow)*2500*86400
        
        ElectronCountsHigh2NoAng = np.sum(dratedTSMHigh2NoAng*dTSMHigh)*2500*86400
        ElectronCountsMid2NoAng = np.sum(dratedTSMMid2NoAng*dTSMMid)*2500*86400
        ElectronCountsLow2NoAng = np.sum(dratedTSMLow2NoAng*dTSMLow)*2500*86400
        
        exclude_prod_High2 = guess_prod * (des_Events_eHigh/ElectronCountsHigh2)**(1/4)
        exclude_prod_Mid2 = guess_prod * (des_Events_eMid/ElectronCountsMid2)**(1/4)
        exclude_prod_Low2 = guess_prod * (des_Events_eLow/ElectronCountsLow2)**(1/4)
        
        exclude_prod_High2NoAng = guess_prod * (des_Events_eHigh_NoAng/ElectronCountsHigh2NoAng)**(1/4)
        exclude_prod_Mid2NoAng = guess_prod * (des_Events_eMid_NoAng/ElectronCountsMid2NoAng)**(1/4)
        exclude_prod_Low2NoAng = guess_prod * (des_Events_eLow/ElectronCountsLow2NoAng)**(1/4)
        
        if (np.min([exclude_prod_High2,exclude_prod_Mid2,exclude_prod_Low2]) 
            >np.min([exclude_prod_High2NoAng,exclude_prod_Mid2NoAng,exclude_prod_Low2NoAng])):
            print("Chi 2: Angular Cut Worse")
        
        coup2 = np.min([exclude_prod_High2,exclude_prod_Mid2,exclude_prod_Low2,
                        exclude_prod_High2NoAng,exclude_prod_Mid2NoAng,exclude_prod_Low2NoAng])
        if coup2 == np.inf:
            coup2 = 1e10
        
        CouplingExclude2[mDM_index,delta_index] = coup2
        
        
        dE1 = E1vals[1:]-E1vals[:-1]
        dE1 = np.append(dE1,dE1[-1])
        
        dsigma1dTSMHigh = dsigmaChi1dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E1vals,TSMvalsHigh,mSM = 9.1e-4,Lambda = 1e10)
        dsigma1dTSMMid = dsigmaChi1dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E1vals,TSMvalsMid,mSM = 9.1e-4,Lambda = 1e10)
        dsigma1dTSMLow = dsigmaChi1dTSMFiniteMass(gchi,gSM,mZ,mDM,delta,E1vals,TSMvalsLow,mSM = 9.1e-4,Lambda = 1e10)
        
        cosTheta1High = (delta**2 + 2*TSMvalsHigh*E1vals + 2*delta*mDM)\
            /(2*TSMvalsHigh * sqrt(E1vals**2 - mDM**2))
        cosTheta1Mid = (delta**2 + 2*TSMvalsMid*E1vals + 2*delta*mDM)\
            /(2*TSMvalsMid * sqrt(E1vals**2 - mDM**2))
        cosTheta1Low = (delta**2 + 2*TSMvalsLow*E1vals + 2*delta*mDM)\
            /(2*TSMvalsLow * sqrt(E1vals**2 - mDM**2))
            
        AngleReq1High = np.heaviside(cosTheta1High - cos(ThetaMaxHigh),1)
        AngleReq1Mid = np.heaviside(cosTheta1Mid - cos(ThetaMaxMid),1)
        AngleReq1Low = np.heaviside(cosTheta1Low - cos(ThetaMaxLow),1)
        
        dratedTSMHigh1 = num_protons * np.sum(dPhi1dE1*dsigma1dTSMHigh*dE1
                                            *AngleReq1High, axis = 1)
        dratedTSMMid1 = num_protons * np.sum(dPhi1dE1*dsigma1dTSMMid*dE1
                                            *AngleReq1Mid, axis = 1)
        dratedTSMLow1 = num_protons * np.sum(dPhi1dE1*dsigma1dTSMLow*dE1
                                            *AngleReq1Low, axis = 1)
        
        dratedTSMHigh1NoAng = num_protons * np.sum(dPhi1dE1*dsigma1dTSMHigh*dE1, axis = 1)
        dratedTSMMid1NoAng = num_protons * np.sum(dPhi1dE1*dsigma1dTSMMid*dE1, axis = 1)
        dratedTSMLow1NoAng = num_protons * np.sum(dPhi1dE1*dsigma1dTSMLow*dE1, axis = 1)
        
        ElectronCountsHigh1 = np.sum(dratedTSMHigh1*dTSMHigh)*2500*86400
        ElectronCountsMid1 = np.sum(dratedTSMMid1*dTSMMid)*2500*86400
        ElectronCountsLow1 = np.sum(dratedTSMLow1*dTSMLow)*2500*86400
        
        ElectronCountsHigh1NoAng = np.sum(dratedTSMHigh1NoAng*dTSMHigh)*2500*86400
        ElectronCountsMid1NoAng = np.sum(dratedTSMMid1NoAng*dTSMMid)*2500*86400
        ElectronCountsLow1NoAng = np.sum(dratedTSMLow1NoAng*dTSMLow)*2500*86400
        
        exclude_prod_High1 = guess_prod * (des_Events_eHigh/ElectronCountsHigh1)**(1/4)
        exclude_prod_Mid1 = guess_prod * (des_Events_eMid/ElectronCountsMid1)**(1/4)
        exclude_prod_Low1 = guess_prod * (des_Events_eLow/ElectronCountsLow1)**(1/4)
        
        exclude_prod_High1NoAng = guess_prod * (des_Events_eHigh_NoAng/ElectronCountsHigh1NoAng)**(1/4)
        exclude_prod_Mid1NoAng = guess_prod * (des_Events_eMid_NoAng/ElectronCountsMid1NoAng)**(1/4)
        exclude_prod_Low1NoAng = guess_prod * (des_Events_eLow/ElectronCountsLow1NoAng)**(1/4)
        
        if (np.min([exclude_prod_High1,exclude_prod_Mid1,exclude_prod_Low1]) 
            >np.min([exclude_prod_High1NoAng,exclude_prod_Mid1NoAng,exclude_prod_Low1NoAng])):
            print("Chi 1: Angular Cut Worse")
        
        coup1 = np.min([exclude_prod_High1,exclude_prod_Mid1,exclude_prod_Low1,
                        exclude_prod_High1NoAng,exclude_prod_Mid1NoAng,exclude_prod_Low1NoAng])
        
        if coup1 == np.inf:
            print("PROBLEM - INFINITY")
            #coup1 = 1e20
        
        
        print("coup 1", coup1, "coup 2",coup2)
        CouplingExclude1[mDM_index,delta_index] = coup1
        

fig, ax = plt.subplots()

CS = ax.contourf(mDMlist,deltaVals,np.log10(np.transpose(CouplingExclude1)))

CB = fig.colorbar(CS)
CB.set_label("$\mathrm{log}_{10}(g_{SM} g_{DM})$")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{\mathrm{DM}}$ [GeV]")
plt.ylabel("$\delta$ [GeV]")

plt.title(source + " ; $m_{Z'}$ = "+str(mZ)+ "GeV")
'''

'''
fig = plt.figure("SK Exclusions No Decay")
if source == "TXS":
    plt.plot(mDMlist,CouplingExclude2,"-",label = "$\delta$ ="+str(delta) +" GeV",c = color)
if source == "NGC":
    plt.plot(mDMlist,CouplingExclude2,"--",label = "$\delta$ ="+str(delta) +" GeV",c = color)


fig = plt.figure("SK Exclusions W Decay")
if source == "TXS":
    plt.plot(mDMlist,CouplingExclude1,"-",label = "$\delta$ ="+str(delta) +" GeV",c = color)
if source == "NGC":
    plt.plot(mDMlist,CouplingExclude1,"--",label = "$\delta$ ="+str(delta) +" GeV",c = color)
'''

'''    
fig = plt.figure("SK Exclusions No Decay")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{\chi}$ [GeV]")
plt.ylabel("$g_{p} g_{\chi}$")
plt.title("No Decay, $m_{Z^{\prime}}$ ="+str(mZ) + "GeV")
plt.legend(fontsize = 5)
plt.xlim([1e-5,1e5])
plt.ylim([1e-7,1e3])


fig = plt.figure("SK Exclusions W Decay")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{\chi}$ [GeV]")
plt.ylabel("$g_{p} g_{\chi}$")
plt.title("Including Decay, $m_{Z^{\prime}}$ ="+str(mZ) + "GeV")
plt.legend(fontsize = 5)
plt.xlim([1e-5,1e5])
plt.ylim([1e-7,1e3])
'''

