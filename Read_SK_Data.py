# -*- coding: utf-8 -*-
"""
Code to read in data from
    SK-Exclusions
"""
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
from numpy.random import randint
import numpy.random as rand

#=================================================================
# Fixed Mediator Mass, different Masses and splittings
#=================================================================
'''
mZ = 500 #GeV
mp = 0.94 #GeV
Inv_GeV_to_cm = 1.98e-14
filename = "SK_Electron_Scattering_mZ_"+str(mZ)+"GeV_w_DIS_Rprod_1000_HighE.csv"

file = open(filename,"r")

line_index = 0
for line in file:
    line_index += 1
    if line_index == 1:
        line = line.split(',')
        try:
            line.remove('\n')
        except:
            nothing = 0
        
        try:
            line.remove(' \n')
        except:
            nothing = 0
        sources = []
        for element in line:
            if element != " " and element != "":
                sources.append(element)
    
    elif line_index == 2:
        line = line.split(',')
        delta_vals = np.array([])
        for element in line:
            try:
                element = element.replace("delta =","")
                element = element.replace("GeV","")
                delta_vals = np.append(delta_vals,float(element))
            except:
                nothing = 0
            if delta_vals[0] == delta_vals[-1] and len(delta_vals) > 1:
                delta_vals = delta_vals[:-1]
                break

file.close()
source_index = -1

fig = plt.figure("Coup1 Exclusions DIS p")
fig = plt.figure("Coup2 Exclusions DIS p")
fig = plt.figure("Cross Sec1 DIS p")
fig = plt.figure("Cross Sec2 DIS p")


styles = ["-","--"]# [':','-.']
colors = ["r","y","g","b","purple"]

for source in sources:
    print(source)
    source_index += 1
    delta_index = -1
    style = styles[source_index]
    for delta in delta_vals:
        delta_index += 1
        color = colors[delta_index]
        mDM_vals = np.array([])
        Coup_vals1 = np.array([])
        Coup_vals2 = np.array([])
        
        file = open(filename,"r")
        line_index = 0
        for line in file:
            line_index += 1
            
            if line_index > 3:
                line = line.split(',')
                
                mDM_vals = np.append(mDM_vals, float(line[source_index * 3*len(delta_vals) + 3*delta_index]))
                Coup_vals1 = np.append(Coup_vals1, float(line[source_index * 3*len(delta_vals) + 3*delta_index+1]))
                Coup_vals2 = np.append(Coup_vals2, float(line[source_index * 3*len(delta_vals) + 3*delta_index+2]))
        
        print(mDM_vals)
        plt.figure("Coup1 Exclusions DIS p")
        plt.plot(mDM_vals,Coup_vals1,style,label = source +" $\delta$ ="+str(round(delta,2))+"GeV",c=color)
        plt.figure("Coup2 Exclusions DIS p")
        plt.plot(mDM_vals,Coup_vals2,style,label = source +" $\delta$ ="+str(round(delta,2))+"GeV",c=color)
        
        mu_DM_p = (mDM_vals*mp)/(mDM_vals + mp)
        sigma0_1 = 4 * Coup_vals1**2 * mu_DM_p**2 /(pi * mZ**4) * Inv_GeV_to_cm**2  #cm^{2}
        sigma0_2 = 4 * Coup_vals2**2 * mu_DM_p**2 /(pi * mZ**4) * Inv_GeV_to_cm**2  #cm^{2}
        
        plt.figure("Cross Sec1 DIS p")
        plt.plot(mDM_vals,sigma0_1,style,label = source +" $\delta$ ="+str(round(delta,2))+"GeV",c=color)
        plt.figure("Cross Sec2 DIS p")
        plt.plot(mDM_vals,sigma0_2,style,label = source +" $\delta$ ="+str(round(delta,2))+"GeV",c = color)

file.close()        

plt.figure("Coup1 Exclusions DIS p")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{\chi}$ [GeV]")
plt.ylabel("$g_{SM} g_{\chi}$")
plt.title("m_{Z'} = "+str(mZ) +" GeV, $\chi_{1}$ exclusions")
plt.legend(fontsize = 4)

plt.figure("Coup2 Exclusions DIS p")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{\chi}$ [GeV]")
plt.ylabel("$g_{SM} g_{\chi}$")
plt.title("m_{Z'} = "+str(mZ) +" GeV, $\chi_{2}$ exclusions")
plt.legend(fontsize = 4)

plt.figure("Cross Sec1 DIS p")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{\chi}$ [GeV]")
plt.ylabel("$\\sigma_{0} = \\frac{4 g_{SM}^2 g_{\chi}^2 \mu_{\chi - p}^2}{\pi m_{Z'}^4}$")
plt.title("m_{Z'} = "+str(mZ) +" GeV $\chi_{1}$ exclusions")
plt.legend(fontsize = 4)       

plt.figure("Cross Sec2 DIS p")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{\chi}$ [GeV]")
plt.ylabel("$\\sigma_{0} = \\frac{4 g_{SM}^2 g_{\chi}^2 \mu_{\chi - p}^2}{\pi m_{Z'}^4}$")
plt.title("m_{Z'} = "+str(mZ) +" GeV $\chi_{2}$ exclusions")
plt.legend(fontsize = 4)     
'''          
                
#==============================================================================
#Fixed Ratios
#==============================================================================
delta_ratio = 0.8
mZ_ratio = 3

DIS = True
Electron = False

source_styles = ["-","--"]

colors = ["r","y","g","b"]
color_number = 0

if DIS:
    DIS_string = "_w_DIS"
    color_number += 2
else:
    DIS_string = "_No_DIS"
    
if Electron:
    part_string = "E"
    color_number += 1
else:
    part_string = "P"
  
color = colors[color_number]
    
filename = "SK_"+part_string+"_Scat_mZ_Ratio_"+str(mZ_ratio)+"_Delta_Ratio"+str(delta_ratio)+DIS_string +'.csv'
file = open(filename,'r')
line_index = 0

for line in file:
    line_index += 1
    line = line.split(',')
    if line_index == 1:
        sources = [line[0],line[3]]
file.close()

use_y = False
use_epsilon = False

if delta_ratio == 0.4 and mZ_ratio == 10:
    plt.figure('y-Vals')
    use_y = True
elif delta_ratio == 0.8 and mZ_ratio == 3:
    plt.figure('epsilon-Vals')
    use_epsilon = True
    alpha_D = 0.5

source_index = -1    
for source in sources:
    source_index += 1
    style = source_styles[source_index]
    mDM_vals = np.array([])
    coup1_vals = np.array([])
    coup2_vals = np.array([])
    
    file = open(filename,'r')
    line_index = 0
    for line in file:
        line_index += 1
        
        if line_index >=3:
            
            line = line.split(',')
            mDM_vals = np.append(mDM_vals,float(line[3*source_index]))
            coup1_vals = np.append(coup1_vals,float(line[3*source_index+1]))
            coup2_vals = np.append(coup2_vals,float(line[3*source_index+2]))
            
        if use_y:
            yvals1 = coup1_vals**2/(4*pi*0.3**2) * (1/mZ_ratio)**4
            yvals2 = coup2_vals**2/(4*pi*0.3**2) * (1/mZ_ratio)**4
        
        if use_epsilon:
            
            eps_vals1 = coup1_vals/(0.3*sqrt(4*pi*alpha_D))
            eps_vals2 = coup2_vals/(0.3*sqrt(4*pi*alpha_D))
            
            
    if use_y:
        plt.figure('y-Vals')
        print('about to plot')
        plt.plot(mDM_vals,yvals1,c=color,linestyle = style,label = source + " SK "+part_string.lower()+DIS_string.replace('_',' '))
    if use_epsilon:
        plt.figure('epsilon_Vals')
        plt.plot(mDM_vals,eps_vals1,c=color,linestyle=style,label = source + " SK "+part_string.lower()+DIS_string.replace('_',' '))
        
    file.close()
 
if use_y:
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
        
    plt.fill_between(mDMexisting,yExisting,100,label = "Existing Bounds",color = "grey",alpha = 0.5)
    plt.plot(mDMthermal,ythermal,label = "Thermal inelastic dark matter",color = "purple")
    
    plt.xlim([1e-3,3e-1])
    plt.ylim([1e-14,1])
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('$m_{\chi}$[GeV]')
    plt.ylabel("y = $\epsilon^2 \\alpha_{D} (m_{\chi}/m_{Z'})^4$")
    plt.legend()

    plt.title('$\delta$ = '+str(round(delta_ratio,1))+"$m_{DM}$ ; $m_{Z'}$ ="+str(mZ_ratio)+"$m_{DM}$")

if use_epsilon:
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

    fig = plt.figure("epsilon_Vals")
    plt.fill_between(mDMexisting,epsilonExisting,1000,label = "Exisiting Bounds", color ="grey", alpha = 0.5)
    #plt.plot(mDMthermal1,epsthermal1,label = "Thermal Relic")
    #plt.plot(mDMthermal2, epsthermal2, label = "Thermal Relic")
    plt.fill_between(mDMthermal1,epsthermal1,epsthermal2interp,label = "Thermal Relic", color = "purple", alpha = 0.5)
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim([3e-2,0.9])
    plt.ylim([1e-4,1e2])

    plt.xlabel('$m_{\chi}$[GeV]')
    plt.ylabel("$\epsilon$")
    plt.legend(fontsize = 5)

    plt.title('$\delta$ = '+str(round(delta_ratio,1))+"$m_{DM}$ ; $m_{Z'}$ ="+str(mZ_ratio)
              +"$m_{DM}$ ; $\\alpha_{D}$ = "+str(alpha_D))

    
    
    