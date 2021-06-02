from Neuron import LIF, ELIF, Adaptive_ELIF
from Synapse import Synapse
import numpy as np
import random
from random import randint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sbn

def current(time, dT, low_y, high_y, frq):
    number_of_dots = int(time * (dT ** (-1))) + 1
    x = np.linspace(0, time, number_of_dots)
    y = np.convolve(randint(0, 100) * np.sin(randint(0, 100) * x), randint(0, 100) * np.cos(randint(0, 100) * x),'same')
    y = np.convolve(randint(0, 100) * np.sin(randint(0, 100) * (x/frq) ), randint(0, 100) * np.cos(randint(0, 100) * (x/frq)),'same')
    y = MinMaxScaler(feature_range=(low_y, high_y)).fit_transform(y.reshape(-1, 1)).reshape(-1)

    out = {}
    for i, t in enumerate(np.arange(0, T, dt)):
        out[t] = y[i]
        
    return out

def constant_current(time, amount):
    curr = {}
    for i , t in enumerate(time):
        curr[t] = amount
    return curr

#---------------------------- Parameters ---------------------------
T = 50
dt = 0.1
time = np.arange(0, T+dt, dt)

I1 = constant_current(time, 20)
I2 = constant_current(time, 21)

# I1 = current(T, dt, 19, 20, 50)
# I2 = current(T, dt, 16, 22, 30)

alpha = 0.5

pre_r = 0.1
pre_c = 0.4
pre_threshold = -55

post_r = 0.1
post_c = 0.4
post_threshold = -55

pre_neuron = LIF(r=pre_r,c=pre_c,threshold=pre_threshold, inputs=I1)
post_neuron = LIF(r=post_r, c=post_c, threshold=post_threshold, inputs=I2)

initial_weight = 1
A_positive = 2
A_negative = -4
tau_positive = 1
tau_negative = 1
max_weight = 20
lamda = 2
betha = 2

synapse = Synapse(pre_neuron, post_neuron, initial_weight, A_positive, A_negative, tau_negative, tau_positive)

time_difference = []
weight_difference = []

for i, t in enumerate(time):

    pre_neuron.compute_potential(t, dt)

    if pre_neuron.last_spike_time == t:
        # print(t)
        pre_post = randint(1,2)
        post_t = float(randint(0, 30)/5)
        if pre_post == 1:
            delta_t = post_t * -1
            w = synapse.w
            new_w = w + synapse.A_negative * np.exp(-1 * np.absolute(delta_t) / synapse.tau_negative)
            synapse.w = new_w
            synapse.weights.append(synapse.w)
            delta_w = synapse.w - w
        else:
            delta_t = post_t
            w = synapse.w
            new_w = synapse.w + synapse.A_positive*np.exp(-1*np.absolute(delta_t)/synapse.tau_positive)
            synapse.w = lamda * ((max_weight - new_w) ** betha)
            synapse.w = new_w
            synapse.weights.append(synapse.w)
            delta_w = synapse.w - w

        time_difference.append(delta_t)
        weight_difference.append(delta_w)


sbn.scatterplot(time_difference,weight_difference)
plt.xlabel("Time difference")
plt.ylabel("Weight difference")
plt.title("STDP")
plt.show()