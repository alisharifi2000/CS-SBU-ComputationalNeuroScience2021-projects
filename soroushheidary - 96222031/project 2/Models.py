import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import numpy as np
import random
import math

class LIF  :

    def __init__(self, I, R=10, tau_m=8, treshold=-45, dspike=5, U_rest=-79, U_reset=-65, ref_period=2, ref_time=0, total_t=100, dt=0.03125, initail_t=0, make_it_pretty=True) : 
        self.I = I
        self.R = (R + random.randint(-3, 3))/1000
        self.U_rest = U_rest - random.randint(-7, 7)
        self.U_reset = U_reset 
        self.tau_m = tau_m
        self.ref_time = ref_time
        self.ref_period = ref_period 
        self.treshold = treshold 
        self.dspike = dspike 
        self.total_t = total_t
        self.dt = dt
        self.initail_t = initail_t
        
        self.tau_to_i = max(1, int(self.ref_period // (self.dt )))
        self.T = np.arange(0, total_t + dt, dt)
        self.Create_U()

        
    
    def Create_U(self) : 
        self.U = np.zeros(len(self.T))
        for i in range(len(self.U)) : 
            self.U[i] = self.U_reset
        self.U[0] = self.U_rest


    def Simulate(self) :
        i=1
        while i <= len(self.T) - 1 : 

            delta_term = (-(self.U[i-1] - self.U_rest) + (self.R * self.I[i-1])  ) / (self.tau_m) * self.dt 
            self.U[i] = self.U[i-1] + delta_term

            self.U[i] = min(self.U[i-1] + delta_term, self.treshold)
            #self.U[i] = max(self.U[i-1] + delta_term, self.U_rest)

            # Spiking 
            if self.U[i] >= self.treshold : 
                self.U[i] += self.dspike
                i += self.tau_to_i
            i+=1



class AELIF(LIF) : 

    def __init__(self, theta_rh=-58, sharpness=1, a=0.01, b=500, tau_w=100, *args, **kwargs) :
        LIF.__init__(self, *args, **kwargs)
        self.theta_rh = theta_rh 
        self.sharpness = sharpness  # why we need this again ?!
        self.a = a * 32
        self.b = b * 32
        self.tau_w = tau_w
        self.derak = 0 
        self.W = np.zeros(len(self.U))    
    
    def Simulate(self) :
        i=1
        while i <= len(self.T) - 1 : 
            expo_term = - (self.U[i-1] - self.U_rest) + self.sharpness * math.exp( (self.U[i-1] - self.theta_rh)/self.sharpness ) 
            self.W[i] = ((self.a * (self.U[i-1] - self.U_rest) - self.W[i-1]) + (self.b * self.tau_w * self.derak)) * (self.dt / self.tau_w) + self.W[i-1]
            delta_term =  ( expo_term + (self.R * self.I[i-1])  - self.R*self.W[i]  ) / (self.tau_m) * self.dt 
            self.derak = 0
            
            self.U[i] = min(self.U[i-1] + delta_term - self.R * self.W[i] + expo_term, self.treshold)
            
            if self.U[i] >= self.treshold : 
                self.derak = 1
                self.W[i+1] = self.W[i] # a bug
                self.U[i] += self.dspike
                for k in range(i + 1, i+self.tau_to_i +1) : 
                    self.W[k] = ((self.a * (self.U[k-1] - self.U_rest) - self.W[k-1])) * (self.dt / self.tau_w) + self.W[k-1]
                i += self.tau_to_i
            i+=1
    
