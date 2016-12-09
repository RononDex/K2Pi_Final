# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:24:15 2016

@author: nsalgo
"""

import numpy as np
import scipy.stats as stats
from threading import Thread
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing.pool import ThreadPool
import threading



data = np.loadtxt('dec_lengths.dat')        #100'000 measurements of decay length of kaons and pions

n=len(data)
tau_pi=4188                                 #average decay length of pions, given
frac_pi=0.84                                #fraction of the pions in the beam

bns = 100    
n_tot, b = np.histogram(data, bins=bns)     #histogram of given data
x=b+(max(b)-min(b))/(2*bns)                 #center of bins
x=x[:-1] 

### Entry point for the estimator
def startmultithreading(enableMultiThreading,N): 
    # If Multi threading is enabled, check how many cores are available on this machine
    if (enableMultiThreading):
        number_of_cores = multiprocessing.cpu_count()
        number_of_threads = number_of_cores
        print("Running on %s cores, using %s threads" % (number_of_cores, number_of_threads))

    # Setup all the variables we need for the simulation
    threads = []    
    liste=[]

    # Use Multithreading to start as many threads as we have cores on this machine
    if (enableMultiThreading):
        
        nThread = int(N / number_of_threads) # The amount of simulation runs every thread should do is n / number_of_threads

        for i in range(number_of_cores):
            # Create the new thread and tell him what function to call with which parameters
            # By giving the same reference to the result list "liste" we ensure that every thread appends 
            # its results directly into this list
            t = Thread(target=runwiththreads, args = (nThread,liste))
            t.start();
            threads.append(t);
            
        # Wait for all threads to finish    
        for i in range(len(threads)):
            if (threads[i].isAlive()):
                threads[i].join()
                
        #return the results
        return liste


### This method that gets called by the threads and estimates nThread amount of decay lengths for tau
def runwiththreads(nThread,tau_k_estimator):
    
    for i in range(nThread):
        dist_pi = stats.expon.rvs(scale=tau_pi, size=int(frac_pi*n))    #exponential distribution with scale == average decay length of pions
        n_pi, b = np.histogram(dist_pi, bins=int(bns), range=[min(data),max(data)])

        n_k=n_tot-n_pi                          #values of histogram of decay length of kaons
        tau_k_est = sum(x*n_k)/sum(n_k)         #average decay length of kaons
        tau_k_estimator.append(tau_k_est)       #list of average decay length of kaons

    
    
N=1000                                          #How many times to calculate average decay length
tau_estimator_liste=startmultithreading(True,N) #List of N average decay length estimations
tau_estimator_array=np.array(tau_estimator_liste)

average_decay_length=np.mean(tau_estimator_array) 
std_k=np.std(tau_estimator_array)               #Standard deviation of distriribution of the tau estimtions
uncertainty_tau=std_k/np.sqrt(N)                #Uncertainity on mean of tau estimations

print(average_decay_length)
print(uncertainty_tau)

plt.figure()                                    #Histogram of distribution of tau estimators
plt.hist(tau_estimator_array, 15)
plt.xlabel('Average decay length of the kaon [m]')
#plt.savefig('hist_tau_estimator3.jpg', dpi=480)
plt.show()
