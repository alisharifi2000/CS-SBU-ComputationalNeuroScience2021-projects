from matplotlib import pyplot as plt

def simulate(data, I_t):
    from math import exp
    tmp = data["U_rest"] - data["U"] + data["delta_t"] * exp((data["U"] - data["theta_rh"]) / data["delta_t"])
    data['U'] += (tmp + data["R"] * I_t) * data["dt"] / data["tau"]
    data["U"] -= data["R"] * data["w_k"] * data["dt"] / data["tau"]
    data["w_k"] += (data["a"] * (data["U"] - data["U_rest"]) - data["w_k"] + data["tau_k"] * data["b"] * int(data["last_fired"])) * data["dt"] / data["tau_k"]

    data['ref_time'] = max(data['ref_time'] - data['dt'], 0)

    if data['last_fired'] or data['ref_time'] != 0:
        data['last_fired'] = False
        data['U'] = data['U_reset']
        return data['U'], data["w_k"]

    if data['U'] >= data['theta']: 
        data['last_fired'] = True
        data['ref_time'] = data['ref_period']
        data['U'] = data['U_spike']

    return data['U'], data["w_k"]


def plot_mv_ms(mv, time_list, top=None, bottom=None):
    plt.plot(time_list, mv)
    plt.ylim(top=top, bottom=bottom)
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    name="[AELIF] Voltage vs Time"
    plt.title(name)
    plt.savefig(name)
    plt.show()

def plot_current(current, time_list):
    plt.plot(time_list, current)
    plt.ylabel('Input current (pA)')
    plt.xlabel('Time (ms)')
    name="[AELIF] Current vs Time"
    plt.title(name)
    plt.savefig(name)
    plt.show()

def plot_internal_current(current, time_list):
    plt.plot(time_list, current)
    plt.ylabel('Adaption current (pA)')
    plt.xlabel('Time (ms)')
    name="[AELIF] Time-Adaption Current"
    plt.title(name)
    plt.savefig(name)
    plt.show()

if __name__ == "__main__":
    func = lambda x: int(10 <= x)*1500
    data = {
        'dt': 0.03125,
        'runtime': 100,
        'func': func,
        'R': .01,
        'tau': 8,
        'theta': 50,
        "U": -70,
        "U_rest": -70,
        "U_reset":-65,
        "U_spike":5,
        "ref_period":2,
        "ref_time":0,
        "theta_rh":-45,
        "delta_t":2,
        "a":0.01,
        "b":500,
        "tau_k":100,
        "w_k": 0,
        "last_fired": False
    }
    time_list=[i * data['dt'] for i in range(int(data['runtime'] // data['dt']))]
    print(data['b'])
    U_over_t = []
    inter_curr = []
    current = [func(t * data['dt']) for t in range(int(data['runtime']//data['dt']))]
    for i in current:
        u, w_k = simulate(data, i)
        U_over_t.append(u)
        inter_curr.append(w_k)
    
    plot_mv_ms(U_over_t, time_list, top=-35, bottom=-80)
    plot_internal_current(inter_curr, time_list)
    plot_current(current, time_list)
