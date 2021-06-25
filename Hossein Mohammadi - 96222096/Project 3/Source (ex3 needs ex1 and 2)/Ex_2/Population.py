from Ex_1.Neurons import *
from numpy.random import normal as np_normal, random as np_random
from numpy import arange, array
import pickle


class Synapse:
    def __init__(self, pre_n, post_n, w=0, delay=0):
        self.pre_n=pre_n
        self.post_n = post_n
        self.w=w
        self.dt=pre_n.dt
        self.synapse_charge=[]
        self.delay=delay
        self.delay_timer=0
        self.delay_multip=1
        self.inject_pulse=False
        self.last_inject_time=[]

    def receive_pulse(self, input):
        self.synapse_charge.append([input*self.w, self.delay+self.dt*self.delay_multip])

    def simulate_synapse(self, stdp_eng=None):
        if not self.synapse_charge: return None
        if self.synapse_charge[0][1]>=self.dt*self.delay_multip:
            self.delay_multip+=1
            return None

        for i, _ in enumerate(self.synapse_charge):
            self.synapse_charge[i][1] -= self.dt * self.delay_multip
        self.delay_multip=1

        # if stdp_eng is not None and self.inject_pulse:
        #     stdp_eng.train_pre_to_post_syn(self)
        # self.inject_pulse=False

        count=0
        for count, (input_u, time_left) in enumerate(self.synapse_charge, start=1):
            if time_left>0: break
            self.post_n.syn_input += input_u * self.post_n.weight_sens
            self.inject_pulse = True
            self.last_inject_time.append(self.pre_n.internal_clock)
        self.synapse_charge=self.synapse_charge[count:]


class FullyConnectedPopulation:
    def __init__(self, J=0.5, stdp_eng=None, delay_range=(0,0), delay_seg=0, *args, **kwargs):
        self.delay_range=delay_range
        if delay_range[0]==delay_range[1]: delay_seg=1
        self.delay_period=(delay_range[1]-delay_range[0])/delay_seg
        self.delay_seg=delay_seg

        self.J=J
        self.neurons = []
        self.stdp_eng=stdp_eng

        self.populate_neurons(*args, **kwargs)
        self.connection_count=delay_seg*len(self.neurons)

        self.create_network()

    def populate_neurons(self, n_type=None, n_config=None, excit_count=None, inhib_count=None, *args, **kwargs):
        for i in range(excit_count):
            self.neurons.append(eval('n_type(is_exc=True, ' + n_config + ')'))
        for i in range(inhib_count):
            self.neurons.append(eval('n_type(is_exc=False, ' + n_config + ')'))

    def create_network(self):
        for pre_neuron in self.neurons:
            for post_neuron in self.neurons:
                if post_neuron != pre_neuron and self.decide_to_connect():
                    self.connect_neurons(pre_neuron, post_neuron)

    def connect_neurons(self, pre_neuron, post_neuron):
        for delay in arange(self.delay_seg):
            syn=Synapse(pre_neuron, post_neuron, self.decide_weight(), delay)
            pre_neuron.post_syn.append(syn)
            post_neuron.pre_syn.append(syn)

    def decide_to_connect(self):
        return True

    def decide_weight(self):
        return self.J / self.connection_count + np_normal(0.0, 0.001)

    def draw_graph(self):
        import networkx as nx
        import pygraphviz
        import matplotlib.pyplot as plt
        ed=[]
        for pre_n_i, pre_neuron in enumerate(self.neurons):
            for synapse in pre_neuron.post_syn:
                ed.append([pre_n_i, self.neurons.index(synapse.post_n), synapse.w*1000//1/1000])
        G = nx.DiGraph()
        G.add_weighted_edges_from(ed)
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args="-Grankdir=LR")
        nx.draw(G,with_labels=True,pos=pos, font_weight='bold')

        edgewidth = [d['weight']*10 for (_, _, d) in G.edges(data=True)]
        # nx.draw_networkx_edges(G, pos, width=edgewidth)
        nx.draw_networkx_edges(G, pos, edge_color=edgewidth)

        plt.show()

    def set_neurons_state(self, input_spike_list, output_spike_list):
        for neuron, to_spike in zip(self.input_neurons+self.output_neurons,
                                    input_spike_list+output_spike_list):
            neuron.syn_input=0;neuron.pre_syn_input=0
            if to_spike: neuron.U=neuron.U_spike+10
            elif neuron.U>neuron.U_reset: neuron.U=neuron.U_reset

    def reset_synapse_charge(self):
        for neuron in self.neurons:
            for post_syn in neuron.post_syn: post_syn.synapse_charge=[]

    def simulate_network_one_step(self, I_t):
        u_history=[]
        i_history=[]
        for neuron in self.neurons:
            inter_U, curr = neuron.simulate_one_step(I_t)
            u_history.append(inter_U)
            i_history.append(curr)

        for neuron in self.neurons:
            for post_s in neuron.post_syn: post_s.simulate_synapse(self.stdp_eng)

        # if self.stdp_eng is not None:
        #     for neuron in self.neurons:
        #         if neuron.last_fired: self.stdp_eng.train_post_to_pre_syn(neuron)

        if neuron.internal_clock%20==0: print(neuron.internal_clock)

        return u_history, i_history


class FixedCouplingPopulation(FullyConnectedPopulation):
    def __init__(self, prob, *args, **kwargs):
        self.prob = prob
        super(FixedCouplingPopulation, self).__init__(*args, **kwargs)

    def decide_to_connect(self):
        return np_random() < self.prob

    def decide_weight(self):
        return self.J/self.connection_count/self.prob + np_normal(0.0, 0.01)


class GaussianFullyConnected(FullyConnectedPopulation):
    def __init__(self, sigma, *args, **kwargs):
        super(GaussianFullyConnected, self).__init__(*args, **kwargs)
        self.sigma = sigma

    def decide_weight(self):
        return np_normal(self.J/self.connection_count, self.sigma/self.connection_count)


# ************* 2 population *************
class FullyConnectedPops(FullyConnectedPopulation):
    def populate_neurons(self, pre_pop=None, post_pop=None, *args, **kwargs):
        self.pre_pop = pre_pop
        self.post_pop = post_pop
        self.neurons = pre_pop.neurons+post_pop.neurons

    def create_network(self):
        for pre_neuron in self.pre_pop.neurons:
            for post_neuron in self.post_pop.neurons:
                self.connect_neurons(pre_neuron, post_neuron)

class FixedCouplingPops(FullyConnectedPops):
    def __init__(self, prob, *args, **kwargs):
        super(FixedCouplingPops, self).__init__(*args, **kwargs)
        self.prob = prob

    def decide_to_connect(self):
        return np_random() < self.prob

    def decide_weight(self):
        return self.J/self.connection_count/self.prob + np_normal(0.0, 0.01)


class GaussianFullyConnectedPops(FullyConnectedPops):
    def __init__(self, sigma, *args, **kwargs):
        self.sigma = sigma
        super(GaussianFullyConnectedPops, self).__init__(*args, **kwargs)

    def decide_weight(self):
        return np_normal(self.J/self.connection_count, self.sigma/self.connection_count)


if __name__ == "__main__":
    from Ex_1.analysis import *
    from Ex_2.analysis import *
    from math import sin

    global dt
    dt=0.03125
    model = GaussianFullyConnectedPops(
        J=6, sigma=1, delay_range=(0,0), delay_seg=1,
        post_pop=FixedCouplingPopulation(
            n_type=AELIF, excit_count=800, inhib_count=200, J=6.5, prob=0.01,
            n_config="dt="+str(dt)+", R=10, tau=8, theta=-40, U_rest=-75, U_reset=-65, U_spike=5, "
                     "weight_sens=1, ref_period=2, ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100"),
        pre_pop=FullyConnectedPopulation(
            n_type=AELIF, excit_count=300, inhib_count=100, J=2.5,
            n_config="dt="+str(dt)+", R=10, tau=8, theta=-40, U_rest=-75, U_reset=-65, U_spike=5, "
                     "weight_sens=1, ref_period=2, ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100")
        )

    # model.draw_graph()
    runtime=20; time_steps=int(runtime//dt)
    curr_func = lambda x: 1515*(sin(x/time_steps*3+1.0)+1) # limited_sin(time_steps)
    u_history=[]; i_history=[]
    plot_current([curr_func(t) for t in range(time_steps)], arange(0,runtime, dt))
    for t in range(time_steps):
        u, cur = model.simulate_network_one_step(curr_func(t*dt))
        u_history.append(u)
        i_history.append(cur)

    # *******************************
    import sys
    sys.setrecursionlimit(100000)

    with open("pop_data1.pickle", "wb") as f:
        pickle.dump(model.pre_pop, f)
    with open("pop_data2.pickle", "wb") as f:
        pickle.dump(model.post_pop, f)

    with open("pop_data1.pickle", "rb") as f:
        model.pre_pop = pickle.load(f)
    with open("pop_data2.pickle", "rb") as f:
        model.post_pop = pickle.load(f)

    plot_mv_ms(array(u_history)[:,1], arange(0,runtime, dt), top=-40, bottom=-80)
    plot_mv_ms(array(u_history)[:,-1], arange(0,runtime, dt), top=-40, bottom=-80)

    plot_raster(*generate_spike_data(model.pre_pop, runtime, dt), runtime, dt, min_t=0)
    plot_raster(*generate_spike_data(model.post_pop, runtime, dt), runtime, dt, min_t=0)
