import itertools
from math import exp


class STDP_engine:
    def __init__(self, phi_p_e=0.01, phi_p_i=0.03, phi_n_e=0.015,
                 phi_n_i=0.045, w_max=1, tau_p=20, tau_n=20, rate=20):
        self.phi_p_e, self.phi_p_i, self.phi_n_e = phi_p_e, phi_p_i, phi_n_e
        self.phi_n_i, self.w_max, self.tau_p, self.tau_n = phi_n_i, w_max, tau_p, tau_n
        self.rate=rate

    def w_func(self, x, pre_w, is_exc):
        is_exc = True if is_exc == 1 else False
        if x >= 0:
            phi = self.phi_p_e if is_exc else self.phi_p_i
            a = phi * (self.w_max - pre_w); tau = -self.tau_p
        else:
            phi = self.phi_n_e if is_exc else self.phi_n_i
            a = -phi * pre_w; tau = self.tau_n
        # print("time delta", x, " | delta", a*exp(x/tau)*10000//1/10000, " | w", pre_w*10000//1/10000)
        return self.rate * a * exp(x / tau)

    def train_pre_to_post_syn(self, synapse):
        t_j = synapse.last_inject_time[-1]
        for t_i in synapse.post_n.t_fired:
            synapse.w += self.w_func(t_i-t_j, synapse.w, synapse.pre_n.is_exc)

    def train_post_to_pre_syn(self, post_neuron):
        t_i = post_neuron.t_fired[-1]
        for pre_synapse in post_neuron.pre_syn:
            for t_j in pre_synapse.last_inject_time:
                pre_synapse.w += self.w_func(t_i-t_j, pre_synapse.w, pre_synapse.pre_n.is_exc)

    # def train(self, neuron):
    #     t_j=neuron.internal_clock; exc=neuron.is_exc
    #     for i, (post_neuron, w) in enumerate(neuron.post_syn):
    #         for t_i in post_neuron.t_fired:
    #             w=neuron.post_syn[i][1]
    #             neuron.post_syn[i][1] = (w+self.w_func(t_i-t_j, w, exc) if w>0 else 0)
    #
    #     t_i=neuron.internal_clock
    #     for i, pre_neuron in neuron.pre_syn:
    #         exc=pre_neuron.is_exc
    #         for t_j in pre_neuron.t_fired:
    #             w=pre_neuron.post_syn[i][1]
    #             pre_neuron.post_syn[i][1] = (w+self.w_func(t_i-t_j, w, exc) if w>0 else 0)

    def shape_of_w(self, min_x, max_x, pre_w, is_exc):
        x=[i/100 for i in range(min_x*100, max_x*100, 3)]
        y=[self.w_func(i, pre_w, is_exc) for i in x]

        from matplotlib import pyplot as plt
        ax = plt.figure().add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.set_ylim(ymin=-max(max(y),-min(y)), ymax=max(max(y),-min(y)))

        plt.plot(x,y)
        plt.show()


if __name__ == "__main__":
    from Ex_1.analysis import random_smooth_array, plot_current, plot_mv_ms, plot_internal_current
    from Ex_2.analysis import *
    from Ex_3.Analysis import *
    from math import sin
    from numpy import random
    from Ex_2.Population import FullyConnectedPopulation

    class CustomModel(FullyConnectedPopulation):
        def create_network(self):
            self.pre_neurons = self.neurons[:-2]
            self.post_neurons = self.neurons[-2:]

            self.input_neurons = self.pre_neurons
            self.output_neurons = self.post_neurons

            for pre_neuron in self.pre_neurons:
                for post_neuron in self.post_neurons:
                    self.connect_neurons(pre_neuron, post_neuron)

    dt=0.03125;test_time=300;runtime=1500+test_time
    time_steps=int(runtime//dt)
    stdp_eng=STDP_engine(phi_p_e=0.01, phi_p_i=0.03,rate=20,
        phi_n_e=0.015, phi_n_i=0.045, w_max=1, tau_p=20, tau_n=20)
    model=CustomModel(n_type=AELIF, excit_count=12, inhib_count=0, J=3,
        delay_range=(0,0), delay_seg=1, stdp_eng=stdp_eng,
        n_config="dt="+str(dt)+", R=10, tau=8, theta=-40, U_rest=-75, U_reset=-65, U_spike=5, "
        "weight_sens=11,ref_period=2, ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100")

    u_history=[]; i_history=[]; w_his=[]; train_delay=50
    freq = lambda x,f: 1 if (x*dt-f*1)%train_delay==0 else 0
    curr_func = lambda x: 0
    for t in range(time_steps):
        if t*dt%train_delay==0:
            model.reset_synapse_charge()
            patter_select=1 if random.random()<0.5 else 0
        if patter_select:
            forced_spike_input=[freq(t,5),freq(t,10),freq(t,5),freq(t,5),freq(t,10),freq(t,5),freq(t,10),0,0,0]
            forced_spike_output=[freq(t,15), freq(t,0)] if model.stdp_eng!=None else []
        else:
            forced_spike_input=[0,0,0,freq(t,5),freq(t,10),freq(t,5),freq(t,10),freq(t,5),freq(t,5),freq(t,10)]
            forced_spike_output=[freq(t,0),freq(t,15)] if model.stdp_eng!=None else []

        model.set_neurons_state(forced_spike_input,forced_spike_output)

        u, cur = model.simulate_network_one_step(curr_func(t))
        w_his.append([o_n.pre_syn[0].w for o_n in model.output_neurons])
        u_history.append(u)
        i_history.append(cur)

        if time_steps-int(test_time//dt)==t: model.stdp_eng=None

    min_t = runtime-120 - test_time
    model.draw_graph()
    stdp_eng.shape_of_w(-60, 60, 0.5, 1)
    plot_current([
        [freq(x,0), 0]+[freq(x,5),freq(x,10),freq(x,5),freq(x,5),freq(x,10),0,0,0,0,0]
        for x in range(time_steps)], arange(0,runtime, dt), max_x=min_t)
    plot_mv_ms(array(u_history), arange(0,runtime, dt), top=-40, bottom=-80, max_x=min_t)
    plot_mv_ms(array(i_history), arange(0,runtime, dt), max_x=0)
    plot_raster(*generate_spike_data(model, runtime, dt), runtime, dt, min_t=0)
    plot_raster(*generate_spike_data(model, runtime, dt), runtime, dt, min_t=min_t)
    w_his=np.array(w_his)
    plot_weight_spike(dt,runtime,model.pre_neurons[0], 0, w_his[:,0], max_x=0)
    plot_weight_spike(dt,runtime,model.pre_neurons[0], 1, w_his[:,1], max_x=0)
