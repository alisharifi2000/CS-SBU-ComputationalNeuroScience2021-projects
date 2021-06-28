def plot_weight_spike(dt,runtime,pre_n, idx_post,w_his, max_x=0):
    from matplotlib import pyplot as plt
    plt.style.use('ggplot')
    a = pre_n.t_fired
    b = []
    c = []
    for i in range(len(a)):
        for j in range(500, 900):
            b.append(a[i])
            c.append(j / 1000)
    plt.scatter(b, c, marker='.', s=5, color='red')

    del a, b, c
    a = pre_n.post_syn[idx_post].post_n.t_fired
    b = []
    c = []
    for i in range(len(a)):
        for j in range(0, 400):
            b.append(a[i])
            c.append(j / 1000)
    plt.scatter(b, c, marker='.', s=5, color='blue')

    k = [a for i, a in enumerate(w_his) if i%int(1/dt)==0]

    plt.plot(k, color='maroon')
    plt.legend(['SynapticChange', 'PreNeuron', 'PostNeuron'], loc='upper left')
    plt.xlim(max_x, runtime)
    plt.show()

if __name__ == "__main__":
    from Ex_1.analysis import random_smooth_array, plot_current, plot_mv_ms, plot_internal_current
    from Ex_2.analysis import *
    from Ex_3.Learning_Alg import *
    from math import sin
    from numpy import random
    from Ex_2.Population import FullyConnectedPopulation

    class CustomModel(FullyConnectedPopulation):
        def create_network(self):
            self.pre_neurons = self.neurons[:-1]
            self.post_neurons = self.neurons[-1:]

            self.input_neurons = self.pre_neurons
            self.output_neurons = self.post_neurons

            for pre_neuron in self.pre_neurons:
                for post_neuron in self.post_neurons:
                    self.connect_neurons(pre_neuron, post_neuron)

        def simulate_network_one_step(self, pre_I_t, post_I_t):
            u_history = []
            i_history = []
            for neuron in self.pre_neurons:
                inter_U, curr = neuron.simulate_one_step(pre_I_t)
                u_history.append(inter_U)
                i_history.append(curr)
            for neuron in self.post_neurons:
                inter_U, curr = neuron.simulate_one_step(post_I_t)
                u_history.append(inter_U)
                i_history.append(curr)

            for neuron in self.neurons:
                for post_s in neuron.post_syn: post_s.simulate_synapse(self.stdp_eng)

            if self.stdp_eng is not None:
                for neuron in self.neurons:
                    if neuron.last_fired: self.stdp_eng.train_post_to_pre_syn(neuron)

            if neuron.internal_clock % 20 == 0: print(neuron.internal_clock)

            return u_history, i_history

    dt=0.03125;runtime=100
    time_steps=int(runtime//dt)
    stdp_eng=STDP_engine(phi_p_e=0.01, phi_p_i=0.03,rate=20,
        phi_n_e=0.015, phi_n_i=0.045, w_max=1, tau_p=20, tau_n=20)
    model=CustomModel(n_type=ELIF, excit_count=2, inhib_count=0, J=1,
        delay_range=(0,0), delay_seg=1, stdp_eng=stdp_eng,
        n_config="dt="+str(dt)+", R=10, tau=8, theta=-45, U_rest=-75, U_reset=-65, U_spike=5, "
        "weight_sens=2,ref_period=2, ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100")

    u_history=[]; i_history=[]; w_his=[]; train_delay=50
    freq = lambda x,f: 1 if (x*dt-f*1)%train_delay==0 else 0
    curr_func1 = lambda x: 1815*(sin(x/time_steps*3.3+1)+2)
    curr_func2 = lambda x: 1815*(sin(x/time_steps*3.3+2)+2) # limited_sin(time_steps)
    for t in range(time_steps):
        u, cur = model.simulate_network_one_step(curr_func1(t),curr_func2(t))
        w_his.append([o_n.pre_syn[0].w for o_n in model.output_neurons])
        u_history.append(u)
        i_history.append(cur)

    min_t = 0
    model.draw_graph()
    plot_mv_ms(array(u_history), arange(0,runtime, dt), top=-40, bottom=-80, max_x=min_t)
    plot_mv_ms(array(i_history), arange(0,runtime, dt), max_x=0)
    plot_raster(*generate_spike_data(model, runtime, dt), runtime, dt, min_t=0)
    plot_raster(*generate_spike_data(model, runtime, dt), runtime, dt, min_t=min_t)
    w_his=np.array(w_his)
    plot_weight_spike(dt,runtime,model.pre_neurons[0], 0, w_his[:,0], max_x=0)
    # plot_weight_spike(dt,runtime,model.pre_neurons[0], 1, w_his[:,1], max_x=0)
