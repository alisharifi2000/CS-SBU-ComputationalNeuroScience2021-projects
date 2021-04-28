from Neuron import Neuron 
import random
import matplotlib.pyplot as plt
import numpy as np 


class Population :
    
    def __init__(self, I_ex,  ex_in_ratio, size=1000, neurons=[], connection_ratio=0.1):
        self.size = size
        self.ex_in_ratio = ex_in_ratio
        self.neurons = neurons
        self.connection_ratio = connection_ratio
        self.I_ex = I_ex
        self.I_in = [0 for i in range(len(I_ex))]
            

    def Neuron_creation(self) : 
        exci_number = int(self.size * self.ex_in_ratio)
        inhi_number = self.size - exci_number

        for i in range(exci_number) : 
            self.neurons.append(Neuron(I = self.I_ex, neuron_type=1))

        for i in range(inhi_number) : 
            self.neurons.append(Neuron(I = self.I_ex, neuron_type=-1))

    def Synaps_creation(self) : 
        chance = int(self.size * self.connection_ratio)
        
        for i in range(len(self.neurons)) : 
            con_list = random.sample(range(0, self.size - 1), chance)
            if i in con_list : 
                con_list.remove(i) 
            self.neurons[i].post_synapses = con_list 


    def Next_tick(self, t) : 
        for neuron in self.neurons : 
            neuron.Simulate_tick(t)

    
    def Adjust_current(self, t) : 
        for neuron in self.neurons : 
            if neuron.just_spiked : 
                for j in neuron.post_synapses : 
                    temp = (0.4  + random.randint(0, 10)/100) * neuron.type
                    self.neurons[j].I[t+1] += temp
                    self.I_in[t+1] += temp


    def Get_activity(self) : 
        activity = [0 for i in range(3200)]
        delta_t = 15
        
        for t in range(delta_t, 3200) : 
            action_sum = 0

            for j in range(len(self.neurons)) :
                temp = np.array(self.neurons[j].spike_history)
                temp = temp[temp <= t]
                temp = temp[temp >= t-delta_t]
                action_sum += len(temp)

            activity[t] = action_sum / delta_t / self.size

        plt.title('Population Activity')
        plt.minorticks_on()
        plt.plot(activity)
        plt.savefig('activity')
        plt.clf()
        

     
    def Raster_plot(self) : 
        raster_list1 = []
        raster_list2 = []
        n_id = 1
        for neuron in self.neurons[ : int(self.ex_in_ratio * self.size)] : 
            for j in neuron.spike_history : 
                raster_list1.append(n_id)
                raster_list2.append(j)
            n_id+=1 


        raster_list3 = []
        raster_list4 = []
        for neuron in self.neurons[int(self.ex_in_ratio * self.size) : ] : 
            for j in neuron.spike_history : 
                raster_list3.append(n_id)
                raster_list4.append(j)
            n_id+=1               

        plt.title('Raster Plot')
        plt.minorticks_on()
        if raster_list1!=[] : 
            plt.plot(raster_list2, raster_list1, 'ko', markersize=0.5, color='orangered')
        if raster_list3!=[] :
            plt.plot(raster_list4, raster_list3, 'ko', markersize=0.5, color='royalblue')
        plt.savefig('Raster_plot')
        plt.clf()
        
    
    



plt.style.use('ggplot')
plt.minorticks_on()

import math
I_ex = [0 for i in range(3201)]
for i in range(3201) : 
    I_ex[i] = (math.cos((i/3201) * math.pi * 2) + 1) * 2000
    #I_ex[i] = (math.cos((i/3201) * math.pi * 2 - 3.14/4) + 1) * 6000



# for i in range(3201) : 
#     if (500 < i < 800) or (2250 < i < 2550) : 
#         I_ex[i] = 5000
#     if 1375 <i< 1675 : 
#         I_ex[i] = 5000
    

plt.plot([i for i in range(3201)], I_ex)
plt.minorticks_on()
plt.title('External Input')
plt.savefig('External I')
#plt.clf()
plt.show()

population = Population(I_ex=I_ex)
print('Population has been created')
population.Neuron_creation()
print('Neurons has been created')
population.Synaps_creation() 
print('Synapses has been created')
for i in range(1, 3200 - 1) : 
    population.Next_tick(i)
    population.Adjust_current(i)
print('Simulations have been completed')
print('Simulation Ended')
population.Get_activity()
population.Raster_plot()


plt.title('Voltage Of A Neuron')
plt.plot(population.neurons[43].T, population.neurons[43].U)
plt.minorticks_on()
#plt.savefig('Voltage of neuron')
#plt.clf()
plt.show()

plt.title('Internal And External Input Of A Neuron')
plt.plot(population.neurons[43].T, population.neurons[43].I)
plt.minorticks_on()
#plt.savefig('Input of neuron')
#plt.clf()
plt.show()

plt.title('Internal Input Of All Neurons Summed')
plt.plot(population.neurons[43].T, population.I_in)
plt.minorticks_on()
#plt.savefig('I_in')
#plt.clf()
plt.show()



############################ PART 2 : 3 populations connected together ##################################
# w13, w23, w31, w32 = 10, 10, 5, 5

# population_in = Population(I_ex=[0 for i in range(3201)], ex_in_ratio=0)
# population_ex1 = Population(I_ex=I_ex, ex_in_ratio=1)
# population_ex2 = Population(I_ex=I_ex, ex_in_ratio=1)

# population_in.Neuron_creation()
# population_in.Synaps_creation()
# population_ex1.Neuron_creation()
# population_ex1.Synaps_creation()
# population_ex2.Neuron_creation()
# population_ex2.Synaps_creation()


# for i in range(1, 3200) : 
#     population_ex1.Next_tick(i)
#     population_ex1.Adjust_current(i) 
#     for neuron in population_in.neurons : 
#         neuron.I[i+1] -= w31 * population_ex1.I_in[i]  

#     population_ex2.Next_tick(i)
#     population_ex2.Adjust_current(i) 
#     for neuron in population_in.neurons : 
#         neuron.I[i+1] -= w32 * population_ex1.I_in[i] 

#     population_in.Next_tick(i)
#     population_in.Adjust_current(i) 
#     for neuron in population_in.neurons : 
#         neuron.I[i+1] += w13 * population_ex1.I_in[i] + w23 * population_ex2.I_in[i]
#         population_in.I_in[i+1] += w13 * population_ex1.I_in[i] + w23 * population_ex2.I_in[i]

    

    
# plt.clf()         
# population_in.Get_activity()
# population_in.Raster_plot()
# plt.clf()
# population_ex1.Get_activity()
# population_ex1.Raster_plot()
# plt.clf()
# population_ex2.Get_activity()
# population_ex2.Raster_plot()
# print('------------------ DONE! -----------------')
###################################################################################################