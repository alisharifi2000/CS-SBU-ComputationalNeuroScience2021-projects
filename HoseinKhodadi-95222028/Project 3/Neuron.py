import numpy as np 
from Synapse import Synapse

class LIF:
    def __init__(self, u_rest=-70, r=5, c=10, threshold=-50, inhibitory=False, inputs={}):
        self.tau = r*c
        self.u_rest = u_rest
        self.R = r
        self.C = c
        self.threshold = threshold
        self.u_spike = 30
        self.potential = self.u_rest
        self.spike_times = []
        self.membrane_potential = []

        self.input = 0
        self.inputs = inputs
        self.inhibitory = inhibitory

        self.post_neurons = []

        self.last_spike_time = -10

    def set_inhibitory(self):
        self.inhibitory = True
        return self

    def compute_potential(self, t, dt):

        current_ = self.inputs.get(t,0)
        u = self.potential - (dt / self.tau) * ((self.potential - self.u_rest ) - (current_*self.R))
        self.membrane_potential.append(u)

        if self.membrane_potential[-1] >= self.threshold:

            self.membrane_potential[-1] = self.threshold+self.u_spike
            self.membrane_potential.append(self.u_rest)

            self.spike_times.append(t)
            self.last_spike_time = t

            self.potential = self.u_rest

            for synapse in self.post_neurons:
                time = np.arange(t, t + synapse.delay * dt, dt)
                for i, tt in enumerate(time):
                    if tt not in synapse.post.inputs.keys():
                        synapse.post.inputs[tt] = 0
                synapse.post.inputs[t+synapse.delay*dt] += ((-1)**int(self.inhibitory) * synapse.w)

        else:
            self.potential += u 

    def compute_input(self, t, alpha):
        input = self.inputs.get(t,0)
        input -= alpha * input
        if input <= 0:
            input = 0
        self.inputs[t] = input

    def epsilon(self, s):
        if s <= 0:
            return 0
        else:
            return (s/self.tau * np.exp(1-s/self.tau))


    def input_reset(self, t, dt, alpha):
        inp = self.inputs.get(t,0)
        if inp != 0:
            self.inputs.pop(t)
            for i in range(1,6):
                self.inputs[i*dt+t ] += inp*alpha

class ELIF(LIF):
    
    def __init__(self, u_rest=-70, r=5, c=10, threshold=-50, inhibitory=False, Delta_T=2, theta_rh=-55, inputs={}):
        super(ELIF, self).__init__( u_rest, r, c, threshold, inputs)
        self.Delta_T = Delta_T
        self.theta_rh = theta_rh


    def compute_potential(self, t, dt):
        current_ = self.inputs.get(t,0)
        u = self.membrane_potential[-1] -  (dt / self.tau * (self.potential - self.u_rest ) - (current_*self.R) - (self.Delta_T*np.exp((self.potential-self.theta_rh)/self.Delta_T)))
        if u >= self.threshold:
            self.membrane_potential.append(self.threshold)
            self.spike_times.append(t)
            self.potential = self.u_rest
            self.membrane_potential.append(self.u_rest)
            self.last_spike_time = t

            for synapse in self.post_neurons:
                time = np.arange(t, t + synapse.delay * dt, dt)
                for i, tt in enumerate(time):
                    if tt not in synapse.post.inputs.keys():
                        synapse.post.inputs[tt] = 0
                synapse.post.inputs[t+synapse.delay*dt] += ((-1)**int(self.inhibitory) * synapse.w)
                # synapse.post.input += ((-1)**int(self.inhibitory) * synapse.w)

        else:
            self.membrane_potential.append(u)
            self.potential = u 



class Adaptive_ELIF(ELIF):
    
    def __init__(self, u_rest=-70, r=5, c=10, threshold=-50,inhibitory=False, Delta_T=2, theta_rh=-55, tau_w=1, a=1, b=1, inputs={}):
        super(Adaptive_ELIF, self).__init__( u_rest, r, c, threshold, Delta_T, theta_rh, inputs)
        self.tau_w = tau_w
        self.w = 0
        self.a = a
        self.b = b

    def compute_potential(self, t, dt):

        current_ = self.inputs.get(t,0)
        w_ = self.w + (dt*self.tau_w) * ( self.a*(self.potential-self.u_rest) - self.w + self.b*self.tau_w*len(self.spike_times))
        u = self.potential -  (dt / self.tau) * ( (self.potential - self.u_rest ) - (current_*self.R) + (self.w*self.R) - (self.Delta_T*np.exp((self.potential-self.theta_rh)/self.Delta_T)))
        
        if u >= self.threshold:
            self.membrane_potential.append(self.threshold)
            self.spike_times.append(t)
            self.potential = self.u_rest
            self.membrane_potential.append(self.u_rest)
            self.last_spike_time = t
            for synapse in self.post_neurons:
                synapse.post.input += ((-1)**int(self.inhibitory) * synapse.w * self.u_rest)

        else:
            self.membrane_potential.append(u)
            self.potential = u

        self.w = w_