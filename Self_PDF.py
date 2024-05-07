"""
Self Calculation of pdfs
"""

import matplotlib
from matplotlib import pyplot as plt
import scienceplots
import numpy as np
from numpy import pi as pi
from numpy import sqrt, cos, sin, exp, log
from matplotlib import ticker, cm
import math
import scipy as sp
from scipy import special as spc
from numpy.random import randint

xVals = np.array([])
QVals = np.array([])
bbarPDFVals = np.zeros((161,37))
cbarPDFVals = np.zeros((161,37))
sbarPDFVals = np.zeros((161,37))
ubarPDFVals = np.zeros((161,37))
dbarPDFVals = np.zeros((161,37))
dPDFVals = np.zeros((161,37))
uPDFVals = np.zeros((161,37))
sPDFVals = np.zeros((161,37))
cPDFVals = np.zeros((161,37))
bPDFVals = np.zeros((161,37))
file = open("CT18NNLO_0000.dat")
index = 0
for line in file:
    #print(line)
    index += 1
    #print('index', index)
    if index < 4:
        continue
    elif index == 4:
        line = line.split("   ")
        line[-1] = line[-1].replace("\n","")
        line = line[1:]
        for j in range(len(line)):
            xVals = np.append(xVals,float(line[j]))
        #print('line',xVals)
    elif index == 5:
        line = line.split("   ")
        line[-1] = line[-1].replace("\n","")
        line = line[1:]
        for j in range(len(line)):
            QVals = np.append(QVals,float(line[j]))
        pdfIndex = 0
    
    elif index > 6:
        xindex = pdfIndex // 37
        Qindex = pdfIndex % 37
        
        line = line.split('  ')
        line = line[1:]
        try:
            line = line.remove(' ')
        except:
            nothing = 0
        line[-1] = line[-1].replace("\n","")
        #print(line)
        if len(line) != 11:
            print("PROBLEM")
        if xindex == 0:
            dx = xVals[0]
            dQ = QVals[0]
        else:
            dx = xVals[xindex] - xVals[xindex - 1]
            dQ = QVals[Qindex] - QVals[Qindex - 1]
        bbarPDFVals[xindex,Qindex] = 5 * float(line[0])
        cbarPDFVals[xindex,Qindex] = 5 * float(line[1])
        sbarPDFVals[xindex,Qindex] = 5 * float(line[2])
        ubarPDFVals[xindex,Qindex] = 5 * float(line[3])
        dbarPDFVals[xindex,Qindex] = 5 * float(line[4])
        dPDFVals[xindex,Qindex] = 5*float(line[5])
        uPDFVals[xindex,Qindex] = 5*float(line[6])
        sPDFVals[xindex,Qindex] = 5*float(line[7])
        cPDFVals[xindex,Qindex] = 5*float(line[8])
        bPDFVals[xindex,Qindex] = 5*float(line[9])
        
        pdfIndex += 1
        
        if pdfIndex == 37*161:
            break

bbarPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(bbarPDFVals)) 
cbarPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(cbarPDFVals)) 
sbarPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(sbarPDFVals)) 
ubarPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(ubarPDFVals)) 
dbarPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(dbarPDFVals))        
dPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(dPDFVals))
uPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(uPDFVals))
sPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(sPDFVals))
cPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(cPDFVals))
bPDFfunc = sp.interpolate.interp2d(xVals,QVals,np.transpose(bPDFVals))

Qmin = QVals[0]
xmin = xVals[0]

#print("Qmin",Qmin)
'''
trialQ = 1000
trialxVals = np.linspace(0.2,1,50)

fig = plt.figure()
plt.plot(trialxVals,dbarPDFfunc(trialxVals,trialQ),label = 'down bar')
plt.plot(trialxVals,ubarPDFfunc(trialxVals,trialQ),label = 'up bar')
plt.plot(trialxVals,sbarPDFfunc(trialxVals,trialQ),label = 'strange bar')
plt.legend()
'''
