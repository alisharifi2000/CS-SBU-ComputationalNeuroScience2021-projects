from brian2 import *
from neurodynex.tools import input_factory, plot_tools
import random
import numpy as np
import matplotlib.pyplot as plt

N                          = 1000
v_rest                     = -70 * mV
v_reset                    = -65 * mV
firing_threshold           = -50 * mV
membrane_resistance        = 10. * Mohm
membrane_time_scale        = 8.  * ms
abs_refractory_period      = 0.2 * ms
duration                   = 100  * ms
PG12                       = 0.2
PG21                       = 0.1
WG12                       = 3  * mV
WG21                       = 2* mV


def get_input_current(amplitude, dt): # returns current timedarray
    inputs = [amplitude*rand() for i in range(int((duration/ms)/(dt/ms)))] * namp
    ta     = TimedArray(inputs, dt=dt)
    return(ta)


def plot_input_current(ta): # plots input current that stimulates neuronal pop.
    plot([i/1000 for i in range(len(ta.values))], ta.values)
    xlabel('time (ms)')
    ylabel('current (namp)')
    show()

input_current = get_input_current(10, 0.1*ms)

def obfuscate(dt):
    tmp = TimedArray([[rand() for j in range(N)] for i in range(int((duration/ms)/(dt/ms)))], dt)
    return(tmp)

obf = obfuscate(0.1*ms)

# LIF equation
eqs = """
    dv/dt =
    ( -(v-v_rest) + membrane_resistance*obf(t,i) * input_current(t) ) / membrane_time_scale : volt (unless refractory)
    """

population = NeuronGroup(N, eqs, threshold='v>=firing_threshold', reset='v=v_reset', refractory=abs_refractory_period, method='linear')
excitatory_population = population[:int(0.8*N)]
excitatory_population.v =  v_rest
inhibtory_population = population[int(0.8*N):]
inhibtory_population.v = 2 * rand() * mV + v_rest

# creating synapses between inhibitory and excitatory populations
G12 = Synapses(excitatory_population, inhibtory_population,  on_pre='v_post += WG12')
G12.connect(p=PG12)

G21 = Synapses(inhibtory_population, excitatory_population, on_pre='v_post -= WG21')
G21.connect(p=PG21)

spike_monitor = SpikeMonitor(population)
state_monitor = StateMonitor(population, ["v"], record=True)


run(duration)


#plot(state_monitor.t/ms, state_monitor.v.T/mV)
#show()
#plot_input_current(input_current)
#plot(spike_monitor.t/ms, spike_monitor.i, '.k')
#xlabel(f'time [ms] ; PG12={PG12}, PG21={PG21}, WG21={WG21}, WG12={WG12}')
#ylabel('neuron index')
#show()


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    show()
    
visualise_connectivity(G12)
