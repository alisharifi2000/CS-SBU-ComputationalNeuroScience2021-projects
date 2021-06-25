import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from Ex_2.Population import *
import pickle


def generate_spike_data(pop, runtime, dt, conv_size = 10):
    spike_history = []
    idx_neuron = []
    neuron_type=[]
    for idx, neuron in enumerate(pop.neurons):
        idx_neuron += [idx for i in neuron.t_fired]

        type=('exc' if neuron.is_exc == 1 else 'inh')
        try:
            if neuron in pop.output_neurons: type="output"
            elif neuron in pop.input_neurons: type="input"
        except: pass
        neuron_type += [type for i in neuron.t_fired]

        spike_history+=neuron.t_fired

    activity = np.bincount(np.array(np.array(spike_history)//dt, dtype = int))
    activity = np.pad(activity, (0, int(runtime//dt-len(activity)+1)), 'constant')[:int(runtime//dt)]
    conv=int(conv_size * (.1 / dt))
    activity = np.convolve(activity, conv*[1/conv], "same")/len(pop.neurons)

    return spike_history, idx_neuron, neuron_type, activity


def plot_raster(spike_history, idx_neuron, neuron_type, activity, runtime, dt, min_t=0):
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1)
    raster = fig.add_subplot(gs[0, 0])
    raster.set_title("Raster plot")
    raster.set(ylabel="Neuron", xlabel="time(S)")
    sns.scatterplot(ax=raster, y=idx_neuron, x=spike_history, hue=neuron_type, marker='.')
    raster.set(xlim=(min_t, runtime))

    pop_activity = fig.add_subplot(gs[1, 0])
    pop_activity.set_title("Population activity")
    pop_activity.set(ylabel="A(t)", xlabel="time(S)")
    sns.lineplot(ax=pop_activity, x=np.arange(0, runtime, dt), y=activity)
    pop_activity.set(ylim=(0, 0.007))
    pop_activity.set(xlim=(min_t, runtime))

    fig.show()
