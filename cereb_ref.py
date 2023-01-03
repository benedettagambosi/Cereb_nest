
import sys
sys.path.append('/home/modelling/Desktop/benedetta/BGs_Cereb_nest_PD/')

import nest
from pathlib import Path
from marco_nest_utils import utils, visualizer as vsl
import pickle
from Cereb import Cereb_class
import time 

load_from_file = True
CORES = 24
VIRTUAL_CORES = 24
RESOLUTION = 0.1
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
Cereb_recorded_names = ['glomerulus', 'purkinje', 'dcn','dcnp', 'io']

len_trial = 3000.
set_time = 1000.
tot_trials= 1
n_rep = 10
if not load_from_file:
    
    for rep in range(n_rep):
        nest.ResetKernel()
        round_time = round(100 * time.time())
        nest.SetKernelStatus({'grng_seed': round_time + 1,
                          'rng_seeds': [round_time + 1+i for i in range(1,25)],
                        #   'rng_seeds': [round_time + 2, round_time + 3, round_time + 4, round_time + 5],
                          'local_num_threads': CORES, 'total_num_virtual_procs': CORES})
        cereb = Cereb_class(nest, hdf5_file_name, n_spike_generators='n_glomeruli')
        CTX = cereb.create_ctxinput(nest, in_spikes="dynamic_poisson")
        recorded_list = [cereb.Cereb_pops[name] for name in Cereb_recorded_names]
        sd_list = utils.attach_spikedetector(nest, recorded_list)

        model_dict = utils.create_model_dictionary(0, Cereb_recorded_names, {**cereb.Cereb_pop_ids}, len_trial,
                                                    sample_time=1., settling_time=set_time,
                                                    trials=tot_trials, b_c_params=[])


        print("Simulating settling time: " + str(set_time) )

        nest.Simulate(set_time)


        for trial in range(tot_trials):
            
            print("Simulating trial: " + str(trial +1) +" di "+ str(tot_trials))
            
            nest.Simulate(len_trial)

        rasters = utils.get_spike_values(nest, sd_list, Cereb_recorded_names)
        fr_stats = utils.calculate_fr_stats(rasters, model_dict['pop_ids'], t_start=set_time)
        with open(f'./cereb_test/rasters_ref_{rep}', 'wb') as pickle_file:
            pickle.dump(rasters, pickle_file)

        with open(f'./cereb_test/model_dic_ref_{rep}', 'wb') as pickle_file:
            pickle.dump(model_dict, pickle_file)
        
else:
    with open(f'./cereb_test/model_dic_ref_0', 'rb') as pickle_file:
            model_dict = pickle.load(pickle_file)

    if n_rep == 1:
        
        with open(f'./cereb_test/rasters_ref_0', 'rb') as pickle_file:
            rasters = pickle.load(pickle_file)

        fr_stats = utils.calculate_fr_stats(rasters, model_dict['pop_ids'], t_start=set_time)
    else:
        raster_list = []
        for rep in range(n_rep):
            
            with open(f'./cereb_test/rasters_ref_{rep}', 'rb') as pickle_file:
                rasters = pickle.load(pickle_file)
            raster_list.append(rasters)

        fr_stats = utils.calculate_fr_stats(raster_list, model_dict['pop_ids'], t_start=set_time, multiple_trials=True)
print(fr_stats['fr']) 
print(fr_stats["CV"])  
Cereb_target = [25.445, 114.332, 46.073]  
fig3, ax3 = vsl.firing_rate_histogram(fr_stats['fr'], fr_stats['name'], CV_list=None, target_fr=Cereb_target, target_CV=None)
with open(f'./cereb_test/ref', 'wb') as pickle_file:
            pickle.dump(fr_stats['fr'][:3], pickle_file)
import matplotlib.pyplot as plt
# fig3, ax3 = vsl.firing_rate_histogram_old(fr_stats['fr'], fr_stats['CV'], fr_stats['name'], 'control', 'cereb')
plt.show()