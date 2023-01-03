
#%%
import pickle as p
import matplotlib.pyplot as plt
import numpy as np
import math


def calculate_fr(raster_list, pop_dim_ids, t_start=0., t_end=None, return_CV_name=False):
    """ Function to evaluate the firing rate and the
    coefficient of variation of the inter spike interval"""
    fr_list = []
    if return_CV_name:
        CV_list = []
        name_list = []
    min_idx = 0  # useful to process neurons indexes

    if t_end == None:
        t_end = np.inf

    for raster in raster_list:
        pop_name = raster['compartment_name']
        pop_dim = pop_dim_ids[pop_name][1] - pop_dim_ids[pop_name][0] + 1
        t_prev = -np.ones(pop_dim)  # to save the last spike time for idx-th neuron
        ISI_list = [[] for _ in range(pop_dim)]  # list of list, will contain the ISI for each neuron
        for tt, idx in zip(raster['times'], raster['neurons_idx'] - pop_dim_ids[pop_name][0] - 1):
            if tt > t_start:  # consider just element after t_start
                if tt < t_end:
                    if t_prev[idx] == -1:  # first spike of the neuron
                        t_prev[idx] = tt
                    else:
                        ISI = (tt - t_prev[idx])  # inter spike interval
                        if ISI != 0:
                            ISI_list[idx] = ISI_list[idx] + [ISI]
                            t_prev[idx] = tt  # update the last spike time
        # we calculate the average ISI for each neuron, comprehends also neurons with fr = 0
        inv_mean_ISI = np.array([1000. / (sum(elem) / len(elem)) if len(elem) != 0 else 0. for elem in ISI_list])
        fr = inv_mean_ISI.mean()
        fr_list = fr_list + [round(fr, 2)]
        if return_CV_name:
            CV_el = np.array([np.array(sublist).std() / np.array(sublist).mean() if len(sublist) != 0 else 0. for sublist in ISI_list])
            CV_list = CV_list + [round(CV_el.mean(), 2)]
            # ISI_array = np.array([item for sublist in ISI_list for item in sublist])  # flat the ISI array
            # CV_list = CV_list + [round(ISI_array.std() / ISI_array.mean(), 2) if len(ISI_array) != 0 else 0.]   # calculate CV between all of the ISI of that population
            name_list = name_list + [raster['compartment_name']]

    if return_CV_name:  # return also ISI array as flatten np.array
        return fr_list, CV_list, name_list
    else:
        return fr_list

# from marco_nest_utils import utils
path = "/home/modelling/Desktop/benedetta/BGs_Cereb_nest_PD/"
name = 'shared_results/complete_580ms_x_101_sol17_both_dopa_EBCC_test34_dcn_io'
# name = "cereb_test"
name1 = "_resting_state_test"
# name1 = "_test_new"
name1 = "_trials_EBCC2_1_LTP_0.000225_LTD_6e-05"
name1=''
f = open(path + name + "/rasters"+name1,"rb")
rster = p.load(f)
f.close()

f = open(path + name + "/model_dic"+name1,"rb")
model = p.load(f)
f.close()

n_trials = model["trials"]
sim_time = model["simulation_time"]
set_time = model["settling_time"]
len_trial = int(sim_time + set_time)
len_trial = int(sim_time)
#%%
fr = calculate_fr(rster, model["pop_ids"], t_start=0., t_end=None, return_CV_name=False)
#%%

iter = model["trials"] 
n_trials = model["trials"]
sim_time = model["simulation_time"]
set_time = model["settling_time"]
len_trial = int(sim_time + set_time)
len_trial = int(sim_time)

g_size = {"purkinje":20., "dcn":10., "glomerulus":20., "io":20.}
cells_id = {"purkinje":[model["pop_ids"]["purkinje"][0],model["pop_ids"]["purkinje"][1]], 
            "dcn":[model["pop_ids"]["dcn"][0],model["pop_ids"]["dcn"][1]], 
            "glomerulus":[model["pop_ids"]["glomerulus"][0],model["pop_ids"]["glomerulus"][0]+300], 
            "io":[model["pop_ids"]["io"][0],model["pop_ids"]["io"][1]]}


shift = set_time
all_spikes = {}
#all_SDF = {}
all_sdf_mean = {}
all_sdf_mean_filt = {}


cells = ["io", "glomerulus","purkinje", "dcn"]
# cells = ["dcn"]
# cells = ["glomerulus"]
for cell in cells:
    for i in range(len(rster)):
        if rster[i]["compartment_name"] == cell:
            pass
            SDF = {}
            sdf_mean = {}
            sdf_mean_filt = {}
            
            evs = rster[i]["neurons_idx"]
            ts = rster[i]["times"]

            spikes = {}

            for i in range(cells_id[cell][0],cells_id[cell][1]):
            # for i in range(model["pop_ids"][cell][0],320): #model["pop_ids"][cell][1]):
                #print(i)
                evs_i = evs == i

                n_evs_i = len(evs[evs_i])
                ts_i =  ts[evs_i]
                spikes[i] = {"n":n_evs_i, "ts" :ts_i}

            
            all_spikes[cell]= spikes

            for trial in range(n_trials):
                sdf_mean[trial] = np.zeros(len_trial)
                for id_cell in all_spikes[cell].keys():
                    t_spikes = all_spikes[cell][id_cell]["ts"]
                    t_spikes_1 = t_spikes[( t_spikes>len_trial*trial+shift) & (t_spikes<len_trial*(1+trial)+shift)]

                    SDF[id_cell] = {}
                    SDF[id_cell][trial] = np.empty(len_trial)
                    for time in range(len_trial):
                        tau_first = time - (t_spikes_1 -len_trial*trial-shift)
                        SDF[id_cell][trial][time] = sum(1/(math.sqrt(2*math.pi)*g_size[cell])*np.exp(-np.power(tau_first,2)/(2*(g_size[cell]**2))))*(10**3)
                    #plt.plot(SDF[id_cell][trial])
                    sdf_mean[trial] += SDF[id_cell][trial]
                    #print(max(SDF[id_cell][trial]))
                #plt.plot(sdf_mean[trial])

                sdf_mean[trial] = sdf_mean[trial]/len(all_spikes[cell].keys())
                sdf_mean_filt[trial] = np.convolve(sdf_mean[trial],np.ones(100))/100
                #SDF[id_cell] = filtered_al

            all_spikes[cell] = spikes
#            all_SDF[cell] = SDF
            all_sdf_mean[cell] = sdf_mean
            all_sdf_mean_filt[cell] =sdf_mean_filt








# %%

plt.plot(all_sdf_mean["purkinje"][30])

#%%
plt.plot(all_sdf_mean["purkinje"][30][150:300])
# %%
import seaborn as sns
iter = model["trials"] 
palette = list(reversed(sns.color_palette("viridis", iter).as_hex()))
print(palette)

# %%
for i in range(0,101):
    #plt.title(cell  + name1)
    plt.plot(all_sdf_mean["glomerulus"][i])#, palette[i])
plt.show()
# %%
