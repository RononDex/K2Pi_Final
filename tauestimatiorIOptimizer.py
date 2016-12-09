# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:55:48 2016

@author: nsalgo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


data = np.loadtxt('dec_lengths.dat')                #100'000 measurements of decay length of kaons and pions
bns = 100    
n, b = np.histogram(data, bins=bns)                 #histogram of data

y=n                                                 #number of decays
x=b+(max(b)-min(b))/200
x=x[:-1]                                            #center of bins

def f(x,A,B,C,D):                                   #convolution of the two exponential distributions
#A = number of kaons in sample 
#B = frac of pions / frac of K 
#C = decay length of pion 
#D = the decay length of the Kaon
    return A * (B * np.exp(-1 * x / C) + np.exp(-1 * x / D) )
    
#param = estimated parameters of the funcrion f
#cov = covariance of  the parameters 
param, cov = scipy.optimize.curve_fit(f,x,y,bounds=([0,0,4187,0], [100000, 0.84/0.16 ,4189,4188]),p0=[16000,5.25,4188, 500])


average_decay_length=param[-1]                      #average decay length of the kaon
uncertainty= np.sqrt(np.diag(cov))[-1]              #uncertainty on the average decay length of  the kaon

print(average_decay_length,uncertainty)
print(param,np.sqrt(np.diag(cov)))

plt.figure()
plt.plot(x,y, label='data')                         #plot of data
plt.plot(x, f(x, *param),label='fitted function')   #plot of function with estimated parameters

plt.xlabel('decay length [m]')
plt.ylabel('number of decays')
plt.legend()
##plt.savefig('PlotTauEstimator.jpg', dpi=480)
plt.show()
