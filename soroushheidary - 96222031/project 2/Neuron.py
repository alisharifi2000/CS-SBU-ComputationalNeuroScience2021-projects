from Models import AELIF
import math


class Neuron(AELIF) : 
    
    def __init__(self, post_synapses=[], neuron_type=1, *args, **kwargs) : 
        AELIF.__init__(self, *args, **kwargs)
        self.spike_history = []
        self.post_synapses = post_synapses
        self.type = neuron_type
        self.rehab_time = 0 
        self.just_spiked = False
        

    def Simulate_tick(self, i) :

        self.just_spiked = False

        if self.rehab_time == 0 : 
            expo_term = - (self.U[i-1] - self.U_rest) + self.sharpness * math.exp( (self.U[i-1] - self.theta_rh)/self.sharpness ) 
            self.W[i] = ((self.a * (self.U[i-1] - self.U_rest) - self.W[i-1]) + (self.b * self.tau_w * self.derak)) * (self.dt / self.tau_w) + self.W[i-1]
            delta_term =  ( expo_term + (self.R * self.I[i-1])  - self.R*self.W[i]  ) / (self.tau_m) * self.dt 
            self.U[i] = min(self.U[i-1] + delta_term, self.treshold + self.dspike)
            self.Check_spike(i)
        
        else : 
            self.rehab_time -= 1
            self.W[i] = ((self.a * (self.U[i-1] - self.U_rest) - self.W[i-1])) * (self.dt / self.tau_w) + self.W[i-1]


        
    def Check_spike(self, i) : 

        if self.U[i] >= self.treshold : 
            self.W[i+1] = self.W[i] 
            self.U[i] += self.dspike
            self.rehab_time += self.tau_to_i
            self.spike_history.append(i)
            self.rehab_time = self.tau_to_i
            self.just_spiked = True
        

   
