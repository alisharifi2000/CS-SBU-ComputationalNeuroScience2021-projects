from matplotlib import pyplot as plt
from Ex_1.Neurons import simulate_with_func, AELIF, LIF, ELIF


def plot_mv_ms(mv, time_list, name="", top=None, bottom=None):
    plt.plot(time_list, mv)
    plt.ylim(top=top, bottom=bottom)
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    if name!="": name=" for "+name
    name="Voltage-Current"+name
    plt.title(name)
    plt.savefig(name)
    plt.show()


def plot_current(current, time_list, name=""):
    plt.plot(time_list, current)
    plt.ylabel('Input current (pA)')
    plt.xlabel('Time (ms)')
    if name!="": name=" for "+name
    name="Time-Current"+name
    plt.title(name)
    plt.savefig(name)
    plt.show()


def plot_internal_current(current, time_list, name=1):
    plt.plot(time_list, current)
    plt.ylabel('Adaption current (pA)')
    plt.xlabel('Time (ms)')
    if name!=1: name=" for "+name
    name="Time-Adaption Current"+name
    plt.title(name)
    # plt.savefig(name)
    plt.show()


def get_freq_vs_current(type, *args, **kwargs):
    current_list = [i for i in range(1000, 8000, 50)]
    freq_list = []
    for const_I in current_list:
        print("checking with current="+str(const_I))
        U_over_t, _, current = simulate_with_func(
            type, 10000, lambda x: const_I, *args, **kwargs)
        freq_list.append(len([0 for a in U_over_t if a > 0]) / 10000.0)

    plt.plot(current_list, freq_list)
    plt.ylabel('Frequency (KHz)')
    plt.xlabel('Input Current')
    plt.show()


def random_smooth_array(l):
    from numpy import random, array, convolve, ones, linspace
    x=linspace(0, 1000, num=l*10)
    y = 0
    result = []
    for _ in x:
        result.append(y)
        y += random.normal(scale=1)
    r_x=10
    random_array=convolve(array(result), ones((r_x,)) / r_x)[(r_x - 1):]
    return lambda x: abs(random_array[int(x*10)])*150


def limited_sin(time_steps):
    from math import sin
    from random import random

    rand_array = random_smooth_array(time_steps)

    a=[]
    for t in range(time_steps):
        x=t/time_steps*22+0.001
        a.append(2*sin(x)/x+sin(x*0.8)+1.5+rand_array(t)/50000)
    return lambda i: a[int(i)]*900


if __name__ == "__main__":
    dt=0.03125; runtime = 100

    # **************************************************
    # config = ["LIF","dt=dt, R=10, tau=8, theta=-45, U_rest=-79, U_reset=-65, U_spike=5"]
    config = ["AELIF","dt=dt, R=10, tau=8, theta=-40, U_rest=-70, U_reset=-65, U_spike=5,"
              "ref_period=2, ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100"]
    # config = ["AELIF","dt=dt, R=10, tau=8, theta=-50, U_rest=-70, U_reset=-65, U_spike=10,"
    #           "ref_period=0, ref_time=0, theta_rh=-58, delta_t=1, a=0.01, b=500, tau_k=100"]
    # config = ["ELIF","dt=dt, R=10, tau=8, theta=-40, U_rest=-70, U_reset=-65, U_spike=5,"
    #           "ref_period=2, ref_time=0, theta_rh=-45, delta_t=2"]

    # **************************************************
    # *****freq*****
    # eval("get_freq_vs_current(type=" + config[0] +","+ config[1]+")")

    # **************************************************
    # *****with current*****
    time_list=[i*dt for i in range(int(runtime//dt))]

    # func=random_smooth_array(len(time_list))

    func=lambda x: (int(10<=x<=20)*2000+int(30<=x<=40)*5000+int(50<=x<=60)*7000)

    # func=lambda x: int(10<=x)*3000

    # from numpy import sin;func=lambda x: 4000*(sin(x)+0.9)

    # **************************************************
    U_over_t, inter_curr, current = eval("simulate_with_func(type=" + config[0] +
                            ", run_time=runtime, curr_func=func," + config[1] + ")")
    func_used=" (constant current 2)"
    plot_mv_ms(U_over_t, time_list, name=config[0]+func_used, top=-35, bottom=-80)
    plot_internal_current(inter_curr, time_list, name=config[0]+func_used)
    plot_current(current, time_list, name=config[0]+func_used)
