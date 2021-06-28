from math import exp
import itertools


class LIF:
    def __init__(self, dt=0.03125, tau=8, theta=-45, R=10, U_rest=-79, U_reset=-65,
                 U_spike=5, ref_time=0, ref_period=0, weight_sens=1, is_exc=True, *args, **kwargs):
        self.dt = dt
        self.tau = tau
        self.theta = theta
        self.R = R/1000
        self.U_rest = U_rest
        self.U_reset = U_reset
        self.U_spike = U_spike
        self.ref_period=ref_period
        self.U = self.U_rest
        self.is_exc = 1 if is_exc else -1

        self.last_fired=False
        self.t_fired=[]
        self.ref_time=ref_time
        self.internal_clock = 0

        self.post_syn = []
        self.pre_syn = []
        self.syn_input = 0
        self.weight_sens = weight_sens

    def change_u(self, I_t):
        self.U+=(self.U_rest - self.U + self.R * I_t) * self.dt / self.tau

    def fire(self):
        self.t_fired.append(self.internal_clock)

        self.last_fired=True
        self.ref_time=self.ref_period
        self.U=self.U_spike

    def simulate_one_step(self, I_t):
        self.internal_clock += self.dt

        self.ref_time = max(self.ref_time-self.dt, 0)

        if self.last_fired or self.ref_time!=0:
            self.last_fired=False
            self.U=self.U_reset
            return self.U, I_t

        self.change_u(I_t)

        # if self.syn_input>0: print(self.syn_input)
        self.U+=self.syn_input
        self.syn_input = 0

        if self.U >= self.theta: self.fire()
        if self.dirac() > 0: self.send_pulse()

        return self.U, I_t

    def send_pulse(self):
        for syn in self.post_syn:
            syn.receive_pulse(self.is_exc*self.dirac())

    def dirac(self):
        return int(self.last_fired)
        # sum=0
        # for t_f in self.t_fired: sum+=exp(-(self.internal_clock-t_f)**4/2)
        # ans = sum * (2*3.14159)**(-1/2)*2.5
        # if ans >= 0.05: return ans
        # return 0


class ELIF(LIF):
    def __init__(self, theta_rh=-58, delta_t=1, *args, **kwargs):
        super(ELIF, self).__init__(*args, **kwargs)
        self.theta_rh = theta_rh
        self.delta_t = delta_t

    def change_u(self, I_t):
        F = self.U_rest - self.U + self.delta_t * exp((self.U-self.theta_rh)/self.delta_t)
        self.U += (F + self.R*I_t) * self.dt / self.tau


class AELIF(ELIF):
    def __init__(self, a=0.01, b=500, tau_k=100, *args, **kwargs):
        super(AELIF, self).__init__(*args, **kwargs)
        self.a = a#*32.25
        self.b = b#*32.25
        self.tau_k=tau_k
        self.w_k=0

    def change_u(self, I_t):
        u = self.U

        self.w_k += (self.a*(u-self.U_rest) - self.w_k + self.tau_k*
                     self.b*self.dirac()) * self.dt / self.tau_k

        F=self.U_rest - u + self.delta_t*exp((u-self.theta_rh)/self.delta_t)
        self.U += (F + self.R*I_t - self.R*self.w_k) * self.dt / self.tau

    def simulate_one_step(self, I_t):
        u, _ = super(AELIF, self).simulate_one_step(I_t)
        return u, self.w_k


def simulate_with_func(n_type, run_time, curr_func, dt, n_config, *args, **kwargs):
    neuron = eval('n_type(dt=dt,' +n_config+ ', *args, **kwargs)')
    U_over_t = []
    inter_curr = []
    current = [curr_func(t*neuron.dt) for t in range(int(run_time//neuron.dt))]
    for i in current:
        u, w_k = neuron.simulate_one_step(i)
        U_over_t.append(u)
        inter_curr.append(w_k)
    return U_over_t, inter_curr, current


if __name__ == "__main__":
    from Ex_1.analysis import *

    dt=0.03125; runtime = 100

    U_over_t, inter_curr, current = simulate_with_func(
        n_type=AELIF, dt=dt, run_time=runtime, curr_func=lambda x: 2500,
        n_config="R=10, tau=8, theta=-40, U_rest=-70, U_reset=-65, U_spike=5, ref_period=2, "
                 "ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100")

    time_list=[i*dt for i in range(int(runtime//dt))]
    plot_mv_ms(U_over_t, time_list, top=-35, bottom=-80, name="AELIF, constant current")
    plot_internal_current(inter_curr, time_list, name="AELIF, constant current")
    plot_current(current, time_list, name="AELIF, constant current")
