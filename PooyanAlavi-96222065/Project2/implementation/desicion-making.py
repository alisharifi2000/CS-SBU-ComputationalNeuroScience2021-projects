from brian2 import *
from neurodynex.tools import input_factory, plot_tools
import random
from numpy import *
import matplotlib.pyplot as plt

N                          = 1000
N_EXCITATORY_1             = 400
N_EXCITATORY_2             = 400
N_INHIBITORY               = 200
v_rest                     = -70 * mV
v_reset                    = -65 * mV
firing_threshold           = -50 * mV
membrane_resistance        = 10. * Mohm
membrane_time_scale        = 8.  * ms
abs_refractory_period      = 0.2 * ms
duration                   = 100  * ms
Pie1                       = 0.4
Pie2                       = 0.1
Pe1i                       = 0.1
Pe1e2                      = 0.2                     
Pe2i                       = 0.2
Pe2e1                      = 0.1
Pe1e1                      = 0.6
Pe2e2                      = 0.1
Wie1                       = 0.1  * mV
Wie2                       = 0  * mV
We1i                       = 0.3  * mV
We2i                       = 0  * mV
We1e2                      = 0  * mV
We2e1                      = 0  * mV
We1e1                      = 0  * mV
We2e2                      = 0  * mV
syn_delay                  = 30 * ms


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

inhibitory_population    = NeuronGroup(N_INHIBITORY  , eqs, threshold='v>=firing_threshold', reset='v=v_reset', refractory=abs_refractory_period, method='linear')
excitatory_population   = NeuronGroup(N_EXCITATORY_2+N_EXCITATORY_1, eqs, threshold='v>=firing_threshold', reset='v=v_reset', refractory=abs_refractory_period, method='linear')

excitatory_population_1 = excitatory_population[:N_EXCITATORY_1]
excitatory_population_2 = excitatory_population[N_EXCITATORY_1:]

inhibitory_population.v   =  v_rest+rand()*mV*2
excitatory_population_1.v =  v_rest+rand()*mV*2
excitatory_population_2.v =  v_rest+rand()*mV*2



"""
conneting populations 
>> inhibitory pop. to both excitatory pop. with different wights
>> excitatory pop.1 to both excitatory pop.2 and inhibitory pop. with different wights
>> excitatory pop.1 to excitatory pop.1
>> excitatory pop.2 to both excitatory pop.1 and inhibitory pop. with different wights
>> excitatory pop.2 to excitatory pop.2
"""

# inhibitory synapses (as source) ####################################################
Gie1 = Synapses(inhibitory_population, excitatory_population_1,  on_pre='v_post -= Wie1',)
Gie1.connect(p=Pie1)
Gie1.delay = syn_delay
Gie2 = Synapses(inhibitory_population, excitatory_population_2,  on_pre='v_post -= Wie2')
Gie2.connect(p=Pie2)
Gie2.delay = syn_delay
# excitatory1 synapses (as source) ####################################################
Ge1i = Synapses(excitatory_population_1, inhibitory_population,  on_pre='v_post += We1i')
Ge1i.connect(p=Pe1i)

Ge1e2 = Synapses(excitatory_population_1, excitatory_population_2,  on_pre='v_post += We1e2')
Ge1e2.connect(p=Pe1e2)

Ge1e1 = Synapses(excitatory_population_1, excitatory_population_1,  on_pre='v_post += We1e1')
Ge1e1.connect(p=Pe1e1)

# excitatory2 synapses (as source) ####################################################
Ge2i = Synapses(excitatory_population_2, inhibitory_population,  on_pre='v_post += We2i')
Ge2i.connect(p=Pe2i)

Ge2e1 = Synapses(excitatory_population_2, excitatory_population_1,  on_pre='v_post += We2e1')
Ge2e1.connect(p=Pe2e1)

Ge2e2 = Synapses(excitatory_population_2, excitatory_population_2,  on_pre='v_post += We2e2')
Ge2e2.connect(p=Pe2e2)


"""
for monitoring excitatory pop.s and inhibitory pop.
"""
exc_spike_monitor = SpikeMonitor(excitatory_population)
exc_state_monitor = StateMonitor(excitatory_population, ["v"], record=True)

inh_spike_monitor = SpikeMonitor(inhibitory_population)
inh_state_monitor = StateMonitor(inhibitory_population, ["v"], record=True)


run(duration)


plot(inh_state_monitor.t/ms, inh_state_monitor.v.T/mV)
show()
plot(exc_state_monitor.t/ms, exc_state_monitor.v.T/mV)
show()
fig = figure(num=0, figsize=(12,8))
fig.suptitle(f'Wie1={str(Wie1)},Wie2={str(Wie2)},We1i={str(We1i)},We2i={str(We2i)},We1e2={str(We1e2)},We2e1={str(We2e1)},We1e1={str(We1e1)},We2e2={str(We2e2)}')

ax1 = subplot2grid((1,2), (0,0))
ax2 = subplot2grid((1,2), (0,1))
ax1.set_title('inhibitory spike monitor')
ax2.set_title('excitatory spike monitor')
ax1.set_ylim(-1, N_INHIBITORY+1)
ax2.set_ylim(-1, N_EXCITATORY_1+N_EXCITATORY_2+1)
ax1.grid(True)
ax2.grid(True)
ax1.plot(inh_spike_monitor.t/ms, inh_spike_monitor.i, '.k')
ax2.plot(exc_spike_monitor.t/ms, exc_spike_monitor.i, '.k')
show()

# plot(inh_spike_monitor.t/ms, inh_spike_monitor.i, '.k')
# show()
# plot(exc_spike_monitor.t/ms, exc_spike_monitor.i, '.k')
# xlabel(f'time [ms] ; PG12={PG12}, PG21={PG21}, WG21={WG21}, WG12={WG12}')
# ylabel('neuron index')
# show()

# plot_input_current(input_current)


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
    
# visualise_connectivity(G12)