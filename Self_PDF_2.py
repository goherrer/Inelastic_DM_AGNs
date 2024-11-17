"""
Self Calculation of pdfs second try
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

xvals = np.array([])
Qvals = np.array([]) #GeV
bbarvals = np.zeros((161,37))
cbarvals = np.zeros((161,37))
sbarvals = np.zeros((161,37))
ubarvals = np.zeros((161,37))
dbarvals = np.zeros((161,37))

gvals = np.zeros((161,37))

dvals = np.zeros((161,37))
uvals = np.zeros((161,37))
svals = np.zeros((161,37))
cvals = np.zeros((161,37))
bvals = np.zeros((161,37))

file = open("CalcPDF.csv")
line_index = 0
x_index = 0
q_index = 0

for line in file:
    line = line.split(',')
    if q_index == 0:
        xvals = np.append(xvals,float(line[0]))
    
    bbarvals[x_index,q_index] = float(line[2])
    cbarvals[x_index,q_index] = float(line[3])
    sbarvals[x_index,q_index] = float(line[4])
    ubarvals[x_index,q_index] = float(line[5])
    dbarvals[x_index,q_index] = float(line[6])
    
    gvals[x_index,q_index] = float(line[7])
    
    dvals[x_index,q_index] = float(line[8])
    uvals[x_index,q_index] = float(line[9])
    svals[x_index,q_index] = float(line[10])
    cvals[x_index,q_index] = float(line[11])
    bvals[x_index,q_index] = float(line[12])
    
    line_index += 1
    x_index += 1
    if x_index == 161:
        x_index = 0
        q_index += 1
        Qvals = np.append(Qvals,float(line[1]))
        

file.close()

Qmin = np.min(Qvals)

dPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,dvals)
uPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,uvals)
sPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,svals)
cPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,cvals)
bPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,bvals)

gPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,gvals)

dbarPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,dbarvals)
ubarPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,ubarvals)
sbarPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,sbarvals)
cbarPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,cbarvals)
bbarPDFfunc = sp.interpolate.RectBivariateSpline(xvals,Qvals,bbarvals)

testQ = 5 #GeV
testXedges = np.logspace(-10,0,100000)
testXvals = np.sqrt(testXedges[1:]*testXedges[:-1])
dx = np.transpose([testXedges[1:]- testXedges[:-1]])

Integrand = np.transpose([testXvals]) * (dPDFfunc(testXvals,testQ)+uPDFfunc(testXvals,testQ)
                                         +sPDFfunc(testXvals,testQ) + cPDFfunc(testXvals,testQ)
                                         +gPDFfunc(testXvals,testQ)
                         +dbarPDFfunc(testXvals,testQ)+ubarPDFfunc(testXvals,testQ)
                         +sbarPDFfunc(testXvals,testQ)+cbarPDFfunc(testXvals,testQ))

tot_up =np.sum(dx*(uPDFfunc(testXvals,testQ)-ubarPDFfunc(testXvals,testQ)))
tot_down =np.sum(dx*(dPDFfunc(testXvals,testQ)-dbarPDFfunc(testXvals,testQ)))
tot_strange = np.sum(dx*(sPDFfunc(testXvals,testQ)-sbarPDFfunc(testXvals,testQ)))
print(np.sum(dx*Integrand))
print(tot_up)
print(tot_down)
print(tot_strange)