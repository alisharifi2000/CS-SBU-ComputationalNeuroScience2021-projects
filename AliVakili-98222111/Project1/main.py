from math import exp
import math
from numpy import arange
from pylab import *

# setup parameters and state variables
T = 100  # total time to simulate (msec)
dt = 0.0125  # simulation time step (msec)
time = arange(0, T + dt, dt)  # time array
t_rest = 0  # initial refractory time
# LIF properties
Fm = []
V_reset = -65.0
V_rest = -79.0
Vm = []  # potential (mV) trace over time
for t in time:
    Vm.append(V_rest)
Vm2 = []  # potential (mV) trace over time
for t in time:
    Vm2.append(V_rest)
Vm3 = []  # potential (mV) trace over time
for t in time:
    Vm3.append(V_rest)
Rm = 0.01  # resistance (kOhm)
Cm = 800.0  # capacitance (uF)
tau_m = 8.0  # time constant (msec)
tau_ref = 4.0  # refractory period (msec)
Vth = -45.0  # spike threshold (mV)
V_spike = 5.0  # spike delta (mV)
flag = False
ref_time = 0
inputCurrent = []  # input current (A)
ferq = []
delta_t = 1.0
theta_rh = -58.0
a = 0.3225
b = 500 * 32.25
tau_k = 100.0
w_k = []
for t in time:
    w_k.append(0.0)
for t in time:
    inputCurrent.append((4000 * (math.sin(t) + 0.9)))

for i, t in enumerate(time):
    if t > t_rest:
        ref_time = max(ref_time - dt, 0)
        Vm[i] = Vm[i - 1] + (V_rest - Vm[i - 1] + Rm * inputCurrent[i]) * dt / tau_m
        if flag or ref_time != 0:
            flag = False
            Vm[i] = V_reset
        if Vm[i] >= Vth:
            flag = True
            Vm[i] = V_spike
            ref_time = 0
plot(time, inputCurrent)
title('Input Current')
ylabel('Input Current(A)')
xlabel('Time (msec)')
show()

plot(time, Vm)
title('Leaky Integrate-and-Fire')
ylabel('Membrane Potential (mV)')
xlabel('Time (msec)')
show()

print('Parameters for figure 1(LIF)')
print('Total Time Frame:', T, '(ms)')
print('dt:', dt, '(ms)')
print('R:', Rm, 'KOhm')
print('V_spike:', V_spike, 'mV')
print('Figure1:LIF(I=4000 * (sin(Time) + 0.9)')
print('________________________________________')

# ELIF


for j, t2 in enumerate(time):
    if t2 > t_rest:
        Vm2[j] = Vm2[j - 1] + (
                (V_rest - Vm2[j - 1] + delta_t * exp((Vm2[j - 1] - theta_rh) / delta_t)) + Rm * inputCurrent[
            j]) * dt / tau_m

plot(time, inputCurrent)
title('Input Current')
ylabel('Input Current(A)')
xlabel('Time (msec)')
show()

plot(time, Vm2)
title('Leaky Integrate-and-Fire')
ylabel('Membrane Potential (mV)')
xlabel('Time (msec)')
show()

print('Parameters for figure 1(LIF)')
print('Total Time Frame:', T, '(ms)')
print('dt:', dt, '(ms)')
print('R:', Rm, 'KOhm')
print('V_spike:', V_spike, 'mV')
print('Figure1:LIF(I=4000 * (sin(Time) + 0.9)')
print('________________________________________')

# AELIF

w_k[0]=0.0
for z, t3 in enumerate(time):
    if Vm3[z - 1] >= Vth:
        flag3 = 1
    else:
        flag3 = 0
    if t3 > t_rest:
        V = Vm3[z - 1]
        Vm3[z] = Vm3[z - 1] - (Rm * w_k[z - 1] * dt / tau_m)
        w_k[z] = w_k[z - 1] + (a * (V - V_rest) - w_k[z - 1] + flag3*tau_k * b) * dt / tau_k
plot(time, inputCurrent)
title('Input Current')
ylabel('Input Current(A)')
xlabel('Time (msec)')
show()

plot(time, Vm3)
title('Leaky Integrate-and-Fire')
ylabel('Membrane Potential (mV)')
xlabel('Time (msec)')
show()

plot(time, w_k)
title('Leaky Integrate-and-Fire')
ylabel('Membrane Potential (mV)')
xlabel('Time (msec)')
show()

print('Parameters for figure 1(LIF)')
print('Total Time Frame:', T, '(ms)')
print('dt:', dt, '(ms)')
print('R:', Rm, 'KOhm')
print('V_spike:', V_spike, 'mV')
print('Figure1:LIF(I=4000 * (sin(Time) + 0.9)')
print('________________________________________')
