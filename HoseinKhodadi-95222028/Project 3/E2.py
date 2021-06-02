from Neuron import LIF, ELIF, Adaptive_ELIF
from Synapse import Synapse
import numpy as np
from numpy import random
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

flag = True
normalization = False
patt_sum = 10
fw = np.zeros((10,2))

pattern1 = [1, 2, 3, 3, 2, 1, 0, 0, 0, 0]
pattern2 = [0, 0, 0, 0, 0, 0, 2, 1, 2, 1]

T = 10000
dt = 0.1
time = np.arange(0, T + dt, dt)
distance = 2
time_Window = 5

mu = 1
sigma = 0.25

C = 0.1
R = 0.4
ther = -55
u_rest = -70

A_positive = 0.001
A_negative = -0.01
tau_positive = 1000
tau_negative = 100

alpha = 0.7

I1 = constant_current(time, 0)

input_layer = []
for i in range(10):
        input_layer.append(LIF(r=R, c=C, threshold=ther, inputs=I1))

output_layer = []
for i in range(2):
    output_layer.append(LIF(r=R, c=C, threshold=-60, inputs=I1))

random_weights = random.normal(mu, sigma, 20)
rw = np.zeros((10,2))
for i in range(10):
    for j in range(2):
        syn = Synapse(input_layer[i], output_layer[j], random_weights[i*2 + j], \
                      A_positive, A_negative, tau_negative, tau_positive)
        input_layer[i].post_neurons.append(syn)
        rw[i,j] = syn.w

final_weights = random_weights
print(rw)
 
first = True
for i, t in enumerate(time):
    if t%distance == 0 and first:
        for ind, input_neuron in enumerate(input_layer):
            if pattern1[ind] != 0:
                input_neuron.spike_times.append(t+pattern1[ind]*dt)
        first = False

    elif t%distance == 0 and not first:
        for ind, input_neuron in enumerate(input_layer):
            if pattern2[ind] != 0:
                input_neuron.spike_times.append(t+pattern2[ind]*dt)
        first = True


for i, t in enumerate(time):

    for neuron in input_layer:
        if t in neuron.spike_times:
            neuron.last_spike_time = t
            for syn in neuron.post_neurons:
                syn.post.input += syn.w * (-1) ** neuron.inhibitory
                if t not in syn.post.inputs.keys():
                    syn.post.inputs[t] = 0
                syn.post.inputs[t] += syn.w * (-1) ** neuron.inhibitory

    output_layer[0].compute_potential(t, dt)
    output_layer[0].compute_input(t, alpha)

    output_layer[1].compute_potential(t, dt)
    output_layer[1].compute_input(t, alpha)

    min = 100
    max = 0
    for ind, pre_neuron in enumerate(input_layer):
        for ind2, synapse in enumerate(pre_neuron.post_neurons):
            t_pre = pre_neuron.last_spike_time
            t_post = synapse.post.last_spike_time
            if t_pre >= 0 and t_post >= 0:
                delta_t = t_post - t_pre
                if delta_t >= 0:
                    w = synapse.w
                    dw = synapse.A_positive * np.exp(-np.absolute(delta_t) / synapse.tau_positive)
                    synapse.w = synapse.w + dw

                else:
                    w = synapse.w
                    dw =  synapse.A_negative * np.exp(-1 * np.absolute(delta_t) / synapse.tau_negative)
                    synapse.w += dw

                if synapse.w > max:
                    max = synapse.w

                if synapse.w < min:
                    min = synapse.w

                synapse.weights.append(synapse.w)
                final_weights[ind * 2 + ind2] = synapse.w
                fw[ind, ind2] = synapse.w

    if flag and t%10==0:
        sum1 = 0
        sum2 = 0
        for i in range(20):
            if i%2 == 0:
                sum1 += final_weights[i]
            else:
                sum2 += final_weights[i]
        x = max - min
        for ind, pre_neuron in enumerate(input_layer):
            for ind2, synapse in enumerate(pre_neuron.post_neurons):
                w = synapse.w
                w = w/patt_sum if normalization else w
                fw[ind, ind2] = w

print()
print(fw)