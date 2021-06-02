import numpy as np


class Synapse:
    def __init__(self, presynaptic, postsynaptic, weight, A_positive=5, A_negative=5, tau_negative=5, tau_positive=5, delay=0):
        self.pre = presynaptic
        self.post = postsynaptic
        self.w = weight
        self.delay = delay

        self.weights = []
        self.weights.append(self.w)
        
        self.A_positive = A_positive
        self.A_negative = A_negative
        self.tau_positive = tau_positive
        self.tau_negative = tau_negative

    def alpha_rate(self, t):
        return np.sum([1 - np.exp(f - t) for f in self.pre.spike_times])

    def STDP(self):
        t_pre = self.pre.last_spike_time
        t_post = self.post.last_spike_time
        time_window = 3
        w = 0
        delta_t = 0
        if t_pre >= 0 and t_post >= 0:
            delta_t = t_post - t_pre
            if delta_t >= 0 and delta_t <= time_window:
                w = self.A_positive * np.exp(-np.absolute(delta_t) / self.tau_positive)

            elif delta_t < 0 and np.absolute(delta_t) <= time_window:
                w = self.A_negative * np.exp(-np.absolute(delta_t) / self.tau_negative)
        
        self.weights.append(w)
        self.w = w

        return w, delta_t