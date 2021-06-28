

!pip3 install brian2

"""#Train"""

from brian2 import *

N = 1
eqs = '''
dv/dt = (I*1.5-v)/tau : 1
I : 1
tau : second
'''

input = NeuronGroup(20, eqs, threshold='v>1', reset='v = 0', method='exact')
input.I = [1, 2, 1, 2, 1, 2, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 2, 1, 0, 1]
input.tau = [1*ms]*20


taum = 5*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
vt = -35*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
gmax = 0.05
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt = (ge * (Ee-v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

neurons = NeuronGroup(2, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='euler')
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )
S.connect()
S.w = 'rand() * gmax'

mon = StateMonitor(S, 'w', record=[0, 1])
s_mon2 = SpikeMonitor(input)
s_mon = SpikeMonitor(neurons)

run(100*second, report='text')

plot(s_mon2.t/ms, s_mon2.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index');

plot(s_mon.t/ms, s_mon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index');

s_mon3 = SpikeMonitor(neurons)

input.I = [1, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 1, 1, 2, 1, 2, 1, 2, 1]
run(100*second, report='text')

plot(s_mon3.t/ms, s_mon3.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index');