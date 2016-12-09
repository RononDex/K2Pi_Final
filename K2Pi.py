import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from threading import Thread
import multiprocessing
from multiprocessing.pool import ThreadPool
import threading

### Creates the rotation matrix to rotate around the Y-axis
### Parameters:
###     alpha: alpha angle, as calculated in SimulateKDecayPoint
def RotationAroundYAxis(alpha):                                     
    return np.array([[np.cos(alpha),0,np.sin(alpha)],[0,1,0],[-np.sin(alpha),0,np.cos(alpha)]])    

### Creates the rotation matrix around the X-Axis
### Parameters:
###     beta: beta angle, as calculated in SimulateKDecayPoint
def RotationAroundXAxis(beta):
    return np.array([[1,0,0],[0,np.cos(beta),-np.sin(beta)],[0,np.sin(beta),np.cos(beta)]])

### Transforms the 4-vectors from the K+ frame into the lab frame using a Lorentz boost
### Parameters:
###     b: Beta factor
###     g: Gamma factor 
def LorentzBoost(b,g):
    return np.array([[g,0,0,b*g],[0,1,0,0],[0,0,1,0],[b*g,0,0,g]])

### Simualtes the decay and determines the point where the K+ decays
### Parameters:
###     Simulation constants as defined at the bottom of this script 
def SimulateKDecayPoint(sx, sy, tau):
    # If no spread, alpha and beta are simply 0
    if sx==0 and sy==0:                                             
        alpha = 0
        beta = 0
    else:
        # Create random generated spreading angles alpha and beta for x-, and y-direction
        alpha = np.array(stats.norm.rvs(loc=0, scale=sx, size=1))   
        beta = np.array(stats.norm.rvs(loc=0, scale=sy, size=1))    

    # Calculate the decay length for the K+ particle (again randomly generated)
    vlen= np.array(stats.expon.rvs(loc=0, scale=tau, size=1))

    # Calculate the x, y and z components for the decay point of the K+
    x0= vlen*np.tan(alpha)*np.cos(beta)/np.sqrt(1+np.tan(alpha)**2 *np.cos(beta)**2)                                
    y0= np.sqrt(vlen**2 -(x0**2))*np.sin(beta)                      
    z0= np.sqrt(vlen**2 -(x0**2))*np.cos(beta)
    dp= np.array([x0,y0,z0])                             
    return  dp, alpha, beta

### Simulates a single K2Pi decay
### Parameters:
###     Simulation constants as defined at the bottom of this script
def SimulateK2PiDecay(E_K_0, E_K_plus, p, b, g, tau):              
    # Theta and Phi are uniformly randomly distributedspread between 0 and pi (respectively 0 and 2pi for phi)
    theta = np.array(stats.uniform.rvs(scale=np.pi, size=1))        
    phi = np.array(stats.uniform.rvs(scale=2*np.pi, size=1))     

    # Convert to Cartesian coordiantes   
    x0= np.sin(theta)*np.cos(phi)                                   
    y0= np.sin(theta)*np.sin(phi)                                  
    z0= np.cos(theta)   

    # Create the momentum 4-vectors from in the K+ rest frame
    P_K_0 = np.array([E_K_0,p*x0,p*y0,p*z0])                        
    P_K_plus = np.array([E_K_plus,p*(-x0),p*(-y0),p*(-z0)])

    # Boost the momentum 4-vectors to the lab frame
    P_lab_0 = np.dot(LorentzBoost(b,g),P_K_0.T)                     
    P_lab_plus = np.dot(LorentzBoost(b,g),P_K_plus.T)
    return P_lab_0, P_lab_plus                                

### Creates the position vector of the decay and the pion momentum vectors rotated to match the Cartesian coordinates of the lab frame
### Parameters:
###     Simulation constants as defined at the bottom of this script
def RotateDecayVectors(sx, sy, tau, E_K_0, E_K_plus, p, b, g): 
    # Decay position and rotation angles                
    dp,alpha,beta = SimulateKDecayPoint(sx, sy, tau) 

    # Pion momentum vectors
    P_lab_0,P_lab_plus = SimulateK2PiDecay(E_K_0, E_K_plus, p, b, g, tau)
    P_lab_0=P_lab_0[1:]
    P_lab_plus=P_lab_plus[1:]    
    
    # Rotate the vectors to the lab frame coordinates
    P_lab_0r = np.dot(RotationAroundXAxis(beta),np.dot(RotationAroundYAxis(alpha),P_lab_0.T))
    P_lab_plusr = np.dot(RotationAroundXAxis(beta),np.dot(RotationAroundYAxis(alpha),P_lab_plus.T))

    # Return the results
    return P_lab_0r, P_lab_plusr, dp

### Simulates n kaon decays
### Paramters:
###     All the simulation constants as defined at the bottom of this script
def SimulateNDecays(sx, sy, tau, E_K_0, E_K_plus, p, b, g, n):        
    P_lab_0 = []
    P_lab_plus = []
    dp = []
    # Simulate every K+ particle seperatly
    for i in range(n):
        decay = RotateDecayVectors(sx, sy, tau, E_K_0, E_K_plus, p, b, g)
        P_lab_0.append(decay[0])
        P_lab_plus.append(decay[1])
        dp.append(decay[2])
    
    # Return the results for the n-simulations as arrays
    return P_lab_0, P_lab_plus, dp

### Calculates the distance from the z-axis (from the sensor)
### Parameters:
###     P_lab_0:    Momentum of the neutral pion in the lab frame
###     P_lab_plus: Momentum of the charged pion in the lab frame
###     dp:         decay point
###     a:          detector position
def HitDistance(P_lab_0, P_lab_plus, dp, a):
    # If behind the detector, return something that is out of its detection range                          
    if float(dp[-1]) >= a:                                              
        return [100,100]

    # If decay happens in front of the detector, calculate the distance from the z-axis when it reaches the position of the detector
    else:
        # Find how many times we have to multiply P_lab_0r with dp to get z=a
        n_0 = float((a-dp[-1])/float(P_lab_0[-1]))                      
        n_plus = float((a-dp[-1])/float(P_lab_plus[-1]))
        # Calculate the distance to the z-Axis (r=(x^2+y^2)^(1/2))
        d_0 = np.sqrt((float(dp[0])+float(n_0*P_lab_0[0]))**2+(float(dp[1])+float(n_0*P_lab_0[1]))**2) 
        d_plus = np.sqrt((float(dp[0])+float(n_plus*P_lab_plus[0]))**2+(float(dp[1])+float(n_plus*P_lab_plus[1]))**2)  
        return [d_0, d_plus] 

### Calculates the successrate for a given detector position in the simulation
### Parameters:
###     P_lab_0:    Momentum of the neutral pion in the lab frame
###     P_lab_plus: Momentum of the charged pion in the lab frame
###     dp:         decay point
###     a:          detector position
###     n:          number of decays, to normalize the result
def successrate(P_lab_0, P_lab_plus, dp, a, n):
    success = 0
    for i in range(n):
        if HitDistance(P_lab_0[i], P_lab_plus[i], dp[i], a)[0] <= 2 and HitDistance(P_lab_0[i], P_lab_plus[i], dp[i], a)[1] <= 2:
            success += 1
    return success/n

### Creates the plot for to visualize the result
### Parameters:
###     a_opt:  Optimal distance
###     SR_max: Max success rate
###     A:      used for X-axis, range for simulation
###     SR:     The result, Y-axis (success rates)
def GraficEvaluation(a_opt, SR_max, A, SR):
    plt.figure()
    plt.plot(A,SR)
    plt.xlim(xmin=a_range[0],xmax=a_range[1])
    plt.ylim(ymin=0, ymax=1)
    plt.plot([a_range[0],a_range[1]], [max(SR),max(SR)],'k:')
    plt.plot([a_opt,a_opt], [0,1],'k:')
    plt.xlabel('detector position [m]')
    plt.ylabel(r'success rate [success/$n_{K+}$]')
    plt.show()

### Starts a single experiment
### Parameters:
###     the simulation parameters (see bottom) + the following 2 parameters:
###     SR: (list where the results (success rates) of the simulation are written into),    
###     i:  (index for SR, which is assigned for this thread, to prevent the threads writing into the same list)
def RunExperiment(sx, sy, E_K_0, E_K_plus, p, b, g, a_range, tau, n, SR, i):

    A = np.linspace(*a_range)

    # Run the simulation
    decay = SimulateNDecays(sx, sy, tau, E_K_0, E_K_plus, p, b, g, n)

    # Unpack the result list
    P_lab_0, P_lab_plus, dp = decay[0], decay[1], decay[2]

    # Insert the results into the list SR and the assigned index "i"
    index = 0
    for a in A:
        if SR[i][index] == 0:
            SR[i][index] = successrate(P_lab_0, P_lab_plus, dp, a, n)
        else:
            SR[i][index] = np.mean([SR[i][index], successrate(P_lab_0, P_lab_plus, dp, a, n)])
        index = index + 1
   

### Entry point for the simulation.
def RunExperimentMultiThreaded(sx, sy, E_K_0, E_K_plus, p, b, g, a_range, tau, n, enableMultiThreading):
    
    # If Multi threading is enabled, check how many cores are available on this machine
    if (enableMultiThreading):
        number_of_cores = multiprocessing.cpu_count()
        number_of_threads = number_of_cores
        print("Running on %s cores, using %s threads" % (number_of_cores, number_of_threads))
    # If Multithreading is disabled, use only 1 thread
    else:
        number_of_threads = 1

    # Setup all the variables we need for the simulation
    A = np.linspace(*a_range)
    threads = []

    # SR holds the successrates calculated by the different simulation from all the cores
    # Its a multidimensional list, SR[0] holds a list of the results of the 1st thread, SR[1], from the 2nd thread, ...
    SR = [ [0] * (len(A)) ] * number_of_cores

    # Use Multithreading to start as many threads as we have cores on this machine
    if (enableMultiThreading):
        # We split the simulation into even parts for all the cores.
        # Basically we run the simulation one time for every core
        # The amount of simulation runs every thread should do is n / number_of_threads
        # This gives an even distribution of the work to do across all the cores
        nThread = int(n / number_of_threads)

        for i in range(number_of_cores):
            # Create the new thread and tell him what function to call with which parameters
            t = Thread(target=RunExperiment, args = (sx, sy, E_K_0, E_K_plus, p, b, g, a_range, tau, nThread, SR, i))
            t.start();
            threads.append(t);

    # If Multithreading turned off, just use the "normal" main thread
    else:
        RunExperiment(sx, sy, E_K_0, E_K_plus, p, b, g, a_range, tau, n, SR, 0)    

    # Wait for all threads to finish
    for i in range(len(threads)):
        if (threads[i].isAlive()):
            threads[i].join()
    
    # Calculate the mean from the different simulations to get one single result
    avgSR = [0] * (len(A));
    for i in range (len(A)):
        values = []
        for j in range(number_of_cores):
            values.append(SR[j][i])
        avgSR[i] = np.mean(values)

    # Find the maximum success rate and its distance
    SR_max = max(avgSR)
    a_opt = 0
    for i in range(len(A)):
        if avgSR[i]==SR_max:
            a_opt = A[i]

    print('Optimal position: ', a_opt)
    print('Max success rate: ', SR_max)
    
    #Returns: optimal position of detector, maximum success rate, successrates data, plot
    return a_opt, SR_max, SR, GraficEvaluation(a_opt, SR_max, A, avgSR)    


### -----------------------------------------------------------------------------------------------------
### Parameters for the simulation
### These numbers were calculated as shown in the report
### -----------------------------------------------------------------------------------------------------
### Constants
E_K_0 = 245.5611565 #MeV            # Energy of neutral pion in the K+ frame
E_K_plus = 248.115779 #MeV          # Energy of a charged pion in the K+ frame
p = 205.138 #MeV/c                  # Impuls for the two pions (identical for both)
b = 0.999978336972366               # Beta factor of the K+ as seen from the lab frame
g = 151.924486603                   # Corresponding Gamma factor of the K+ as seen from the lab frame
tau = 576.21338881                  # Average distance traveled when a K+ particle decays in meters (obtained from LifeK)
uncertainty_tau=2.4                 # The calculated uncertainty on the value of tau for K+
sx = 1e-3			                # Standard deviation for the x-Angle in rad
sy = 1e-3   			            # Standard deviation for the y-Angle in rad

### Simulation parameters
n = 100000                          # Amount of Kaons that are used in this simulation
a_range = [0,500,1000]  		    # Start point, end point and resolution in meters for the simulation
enableMultiThreading = True         # Set to true to enable multithraeding, false to disable it


# Start the simulation with the given parameters
a_opt, SR_max, SR, f = RunExperimentMultiThreaded(sx, sy, E_K_0, E_K_plus, p, b, g, a_range, tau, n, enableMultiThreading)
