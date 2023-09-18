
pass 
import sys
sys.path.append('/home/modelling/Desktop/benedetta/BGs_Cereb_nest_PD/')
import nest
from pathlib import Path
from marco_nest_utils import utils
import pickle
from Cereb import Cereb_class
import numpy as np 

CORES = 24
VIRTUAL_CORES = 4
RESOLUTION = 1.
run_on_vm = False

# set number of kernels
nest.ResetKernel()
nest.SetKernelStatus({"total_num_virtual_procs": CORES, "resolution": RESOLUTION})
nest.set_verbosity("M_ERROR")  # reduce plotted info
MODULE_PATH = str(Path.home()) + '/nest/lib/nest/ml_module'
nest.Install(MODULE_PATH)  # Import my_BGs module
MODULE_PATH = str(Path.home()) + '/nest/lib/nest/cerebmodule'
nest.Install(MODULE_PATH)  # Import CerebNEST

hdf5_file_name = "Cereb_nest/scaffold_full_IO_400.0x400.0_microzone.hdf5"



cortex_type = "no"
LTP = 0.0002
LTD = -0.02
tot_trials = 1


cereb = Cereb_class(nest, hdf5_file_name, cortex_type, n_spike_generators='n_glomeruli',
                mode='external_dopa', experiment='no', dopa_depl=0, LTD=LTD, LTP=LTP)

Cereb_recorded_names = ['glomerulus', 'purkinje', 'dcn', 'io']

recorded_list = [cereb.Cereb_pops[name] for name in Cereb_recorded_names]
sd_list = utils.attach_spikedetector(nest, recorded_list)


baseline = 0
len_trial = 2000.


ct = cereb.create_ctxinput(nest, pos_file=hdf5_file_name, in_spikes=cortex_type, 
n_spike_generators='n_glomeruli',
                    experiment='no', tot_trials = tot_trials, len_trial = len_trial)


model_dict = utils.create_model_dictionary(0, Cereb_recorded_names, {**cereb.Cereb_pop_ids}, len_trial,
                                                sample_time=1., settling_time=0.,
                                                trials=tot_trials, b_c_params=[])


#nest.Simulate(set_time)


IO_id = cereb.Cereb_pops['io']
for trial in range(tot_trials):
    
    '''
    # CS_spk = np.around(np.linspace(CS["start"]+ set_time +(trial*len_trial),CS["stop"]+ set_time +(trial*len_trial),22), decimals=1)
    # CS_stim = nest.Create("spike_generator", len(glom_id), {"spike_times":CS_spk})

    # CS_stim = nest.Create("poisson_generator", len(glom_id), {"start":500.+(trial*len_trial), "stop":760.+(trial*len_trial), "rate":36.})
    #nest.Connect(CS_stim, glom_id, "one_to_one")
    '''

    
    US_spk = np.around(np.linspace(300. +(trial*len_trial),350.  +(trial*len_trial),int(500.*1000/(50.))))
    US_stim = nest.Create("spike_generator", len(IO_id), {"spike_times":US_spk})
    
    # US_stim = nest.Create("poisson_generator", len(IO_id), {"start":750.+(trial*len_trial), "stop":760.+(trial*len_trial), "rate":200.})
    nest.Connect(US_stim, IO_id, "all_to_all", {"receptor_type": 1, "delay":1.,"weight":10.})
    
    
    print("Simulating trial: " + str(trial +1) +" di "+ str(tot_trials))
    nest.Simulate(len_trial)

    
rasters = utils.get_spike_values(nest, sd_list, Cereb_recorded_names)
with open(f'./cereb_test/rasters_resting_state_test', 'wb') as pickle_file:
    pickle.dump(rasters, pickle_file)


with open(f'./cereb_test/model_dict_resting_state_test', 'wb') as pickle_file:
    pickle.dump(model_dict, pickle_file)