# -*- coding: utf-8 -*-
"""
"""

__author__ = 'marco'

import numpy as np
import h5py
from copy import deepcopy

# cortex_type = "EBCC2"
# LTP = 0.000
# LTD = -0.5
tot_trials = 30

pf_pc = 0.4
ratio = 1/18.67
pc_dcn = 0.55
pc_dcnp = 0.03
ratio_pc_dcn = 45.5/26
ratio_pc_dcnp = 11.5/26
# Synapse parameters: in E-GLIF, 3 synaptic receptors are present: the first is always associated to exc, the second to inh, the third to remaining synapse type
Erev_exc = 0.0  # [mV]	#[Cavallari et al, 2014]
Erev_inh = -80.0  # [mV]
tau_exc = {'golgi': 0.23, 'granule': 5.8, 'purkinje': 1.1, 'basket': 0.64, 'stellate': 0.64, 'dcn': 1.0, 'dcnp': 3.64,
           'io': 1.0}  # tau_exc for pc is for pf input; tau_exc for goc is for mf input; tau_exc for mli is for pf input
tau_inh = {'golgi': 10.0, 'granule': 13.61, 'purkinje': 2.8, 'basket': 2.0, 'stellate': 2.0, 'dcn': 0.7, 'dcnp': 1.14,
           'io': 60.0}
tau_exc_cfpc = 0.4
tau_exc_pfgoc = 0.5
tau_exc_cfmli = 1.2

# Single neuron parameters:
neuron_param = {'golgi': {'t_ref': 2.0, 'C_m': 145.0,'tau_m': 44.0,'V_th': -55.0,'V_reset': -75.0,'Vinit': -62.0,'E_L': -62.0,'Vmin':-150.0,
                         'lambda_0':1.0, 'tau_V':0.4,'I_e': 16.214,'kadap': 0.217,'k1': 0.031, 'k2': 0.023,'A1': 259.988,'A2':178.01,
                         'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['golgi'], 'tau_syn2': tau_inh['golgi'], 'tau_syn3': tau_exc_pfgoc},
               'granule': {'t_ref': 1.5, 'C_m': 7.0,'tau_m': 24.15,'V_th': -41.0,'V_reset': -70.0,'Vinit': -62.0,'E_L': -62.0,'Vmin': -150.0,
                           'lambda_0':1.0, 'tau_V':0.3,'I_e': -0.888,'kadap': 0.022,'k1': 0.311, 'k2': 0.041,'A1': 0.01,'A2':-0.94,
                           'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['granule'], 'tau_syn2': tau_inh['granule'], 'tau_syn3': tau_exc['granule']},
               'purkinje': {'t_ref': 0.5, 'C_m': 334.0,'tau_m': 47.0,'V_th': -43.0,'V_reset': -69.0,'Vinit': -59.0,'E_L': -59.0,
                            'lambda_0':4.0, 'tau_V':3.5,'I_e': 742.54,'kadap': 1.492,'k1': 0.1950, 'k2': 0.041,'A1': 157.622,'A2':172.622,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['purkinje'], 'tau_syn2': tau_inh['purkinje'], 'tau_syn3': tau_exc_cfpc},
               'basket': {'t_ref': 1.59, 'C_m': 14.6,'tau_m': 9.125,'V_th': -53.0,'V_reset': -78.0,'Vinit': -68.0,'E_L': -68.0,
                          'lambda_0':1.8, 'tau_V':1.1,'I_e': 3.711,'kadap': 2.025,'k1': 1.887, 'k2': 1.096,'A1': 5.953,'A2':5.863,
                          'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['basket'], 'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
               'stellate': {'t_ref': 1.59, 'C_m': 14.6,'tau_m': 9.125,'V_th': -53.0,'V_reset': -78.0,'Vinit': -68.0,'E_L': -68.0,
                            'lambda_0':1.8, 'tau_V':1.1,'I_e': 3.711,'kadap': 2.025,'k1': 1.887, 'k2': 1.096,'A1': 5.953,'A2':5.863,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['basket'], 'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
            #bsb
               'dcn': {'t_ref': 1.5, 'C_m': 142.0,'tau_m': 33.0,'V_th': -36.0,'V_reset': -55.0,'Vinit': -45.0,'E_L': -45.0,
                       'lambda_0':3.5, 'tau_V':3.0,'I_e': 185.0,'kadap': 0.408,'k1': 0.697, 'k2': 0.047,'A1': 13.857,'A2':3.477,
                       'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc},#'tau_syn1': tau_exc['dcn'], 'tau_syn2': tau_inh['dcn'], 'tau_syn3': tau_exc['dcn']},
               'dcnp': {'t_ref': 3.0, 'C_m': 56.0,'tau_m': 56.0,'V_th': -39.0,'V_reset': -55.0,'Vinit': -40.0,'E_L': -40.0,
                        'lambda_0':0.9, 'tau_V':1.0,'I_e': 2.384,'kadap': 0.079,'k1': 0.041, 'k2': 0.044,'A1': 176.358,'A2':176.358,
                        'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': 3.64, 'tau_syn2': 1.14},# 'tau_syn3': tau_exc['dcnp']},
            #     'dcn': {'t_ref': 0.8, 'C_m': 142.0,'tau_m': 33.0,'V_th': -36.0,'V_reset': -55.0,'Vinit': -45.0,'E_L': -45.0,
            #            'lambda_0':3.5, 'tau_V':3.0,'I_e': 75.385,'kadap': 0.408,'k1': 0.697, 'k2': 0.047,'A1': 13.857,'A2':3.477,
            #            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['dcn'], 'tau_syn2': tau_inh['dcn'], 'tau_syn3': tau_exc['dcn']},
            #    'dcnp': {'t_ref': 0.8, 'C_m': 56.0,'tau_m': 56.0,'V_th': -39.0,'V_reset': -55.0,'Vinit': -40.0,'E_L': -40.0,
            #             'lambda_0':0.9, 'tau_V':1.0,'I_e': 2.384,'kadap': 0.079,'k1': 0.041, 'k2': 0.044,'A1': 176.358,'A2':176.358,
            #             'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['dcnp'], 'tau_syn2': tau_inh['dcnp'], 'tau_syn3': tau_exc['dcnp']},
               'io': {'t_ref': 1.0, 'C_m': 189.0,'tau_m': 11.0,'V_th': -35.0,'V_reset': -45.0,'Vinit': -45.0,'E_L': -45.0,
                      'lambda_0':1.2, 'tau_V':0.8,'I_e': -18.101,'kadap': 1.928,'k1': 0.191, 'k2': 0.091,'A1': 1810.93,'A2':1358.197,
                      'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['io'], 'tau_syn2': tau_inh['io'], 'tau_syn3': tau_exc['io']},
                'death_purkinje': {'t_ref': 0.5, 'C_m': 1000.0, 'tau_m': 47.0, 'V_th': 100.0, 'V_reset': -80.0, 'Vinit': -80.0,
                 'E_L': -80.0,
                 'lambda_0': 4.0, 'tau_V': 3.5, 'I_e': 0., 'kadap': 0., 'k1': 1., 'k2': 1., 'A1': 0.,
                 'A2': 0.,
                 'E_rev1': Erev_inh, 'E_rev2': Erev_inh, 'E_rev3': Erev_inh, 'tau_syn1': tau_exc['purkinje'],
                 'tau_syn2': tau_inh['purkinje'], 'tau_syn3': tau_exc_cfpc},}


# Connection weights
# conn_weights = {'pc_dcn': 0.4, 'pc_dcnp': 0.12, 'pf_bc': 0.015, 'pf_goc': 0.05,'pf_pc': pf_pc*ratio, # 0.007, \
#                 'pf_sc': 0.015, 'sc_pc': 0.3, 'aa_goc': 1.2, 'aa_pc': 0.7, 'bc_pc': 0.3, 'dcnp_io': 3.0, 'gj_bc': 0.2, 'gj_sc': 0.2, 'glom_dcn': 0.05,\
#                 'glom_goc': 1.5, 'glom_grc': 0.15, 'goc_glom': 0.0, 'gj_goc': 0.3,'goc_grc': 0.6, 'io_dcn': 0.1, 'io_dcnp': 0.2,\
#                 'io_bc': 1.0,'io_sc': 1.0, 'io_pc': 10.0, }
# conn_weights = {'pc_dcn': pc_dcn/ratio_pc_dcn, 'pc_dcnp': pc_dcnp/ratio_pc_dcnp, 'pf_bc': 0.015, 'pf_goc': 0.05,'pf_pc': pf_pc*ratio, \
#                 'pf_sc': 0.015, 'sc_pc': 0.3, 'aa_goc': 1.2, 'aa_pc': 0.7, 'bc_pc': 0.3, 'dcnp_io': 3.0, 'gj_bc': 0.2, 'gj_sc': 0.2, 'glom_dcn': 0.05,\
#                 'glom_goc': 1.5, 'glom_grc': 0.15, 'goc_glom': 0.0, 'gj_goc': 0.3,'goc_grc': 0.6, 'io_dcn': 0.1, 'io_dcnp': 0.2,\
#                 'io_bc': 1.0,'io_sc': 1.0, 'io_pc': 40.0, }
conn_weights = {'pc_dcn': pc_dcn/ratio_pc_dcn, 'pc_dcnp': pc_dcnp/ratio_pc_dcnp, 'pf_bc': 0.015, 'pf_goc': 0.05,'pf_pc': pf_pc*ratio, \
                'pf_sc': 0.015, 'sc_pc': 0.3, 'aa_goc': 1.2, 'aa_pc': 0.7, 'bc_pc': 0.3, 'dcnp_io': 3.0, 'gj_bc': 0.2, 'gj_sc': 0.2, 'glom_dcn': 0.05,\
                'glom_goc': 1.5, 'glom_grc': 0.15, 'goc_glom': 0.0, 'gj_goc': 0.3,'goc_grc': 0.6, 'io_dcn': 0.1, 'io_dcnp': 0.2,\
                'io_bc': 1.0,'io_sc': 1.0, 'io_pc': 10.0, } #350.

# Connection delays
conn_delays = {'aa_goc': 2.0, 'aa_pc': 2.0, 'bc_pc': 4.0, 'dcnp_io': 20.0, 'gj_bc': 1.0, 'gj_sc': 1.0, 'glom_dcn': 4.0,
               'glom_goc': 4.0, 'glom_grc': 4.0, 'goc_glom': 0.5, 'gj_goc': 1.0, 'goc_grc': 2.0, 'io_dcn': 4.0, 'io_dcnp': 5.0,
               'io_bc': 70.0,'io_sc': 70.0, 'io_pc': 4.0, 'pc_dcn': 4.0, 'pc_dcnp': 4.0, 'pf_bc': 5.0, 'pf_goc': 5.0,'pf_pc': 5.0,
               'pf_sc': 5.0, 'sc_pc':5.0}

sd_iomli = 10.0          # IO-MLI delayes are set as normal distribution to reproduce the effect of spillover-based transmission
min_iomli = 40.0

# Connection receptors
conn_receptors = {'aa_goc': 3, 'aa_pc': 1, 'bc_pc': 2, 'dcnp_io': 2, 'gj_bc': 2, 'gj_sc': 2, 'glom_dcn': 1,
               'glom_goc': 1, 'glom_grc': 1, 'goc_glom': 1, 'gj_goc': 2, 'goc_grc': 2, 'io_dcn': 1, 'io_dcnp': 1,
               'io_bc': 3,'io_sc': 3, 'io_pc': 3, 'pc_dcn': 2, 'pc_dcnp': 2, 'pf_bc': 1, 'pf_goc': 3,'pf_pc': 1,
               'pf_sc': 1, 'sc_pc': 2}

# Receiver plastic (name of post-synaptic neurons for heterosynaptic plastic connections)
receiver = {'pf_pc': 'purkinje', 'pf_bc': 'basket', 'pf_sc': 'stellate', 'glom_dcn': 'dcn', "io_bc":"basket","io_pc":"purkinje", "io_sc":"stellate"}

# Plasticity parameters
LTD_PFPC = -0.02
LTP_PFPC = 0.002


class Cereb_class:
    def __init__(self, nest, hdf5_file_name, cortex_type = "", n_spike_generators='n_glomeruli',
                 mode='external_dopa', experiment='active', dopa_depl=0, LTD=LTD_PFPC, LTP=LTP_PFPC):
        # create Cereb neurons and connections
        # Create a dictionary where keys = nrntype IDs, values = cell names (strings)
        # Cell type ID (can be changed without constraints)
        self.cell_type_ID = {'golgi': 1,
                             'glomerulus': 2,
                             'granule': 3,
                             'purkinje': 4,
                             'basket': 5,
                             'stellate': 6,
                             'dcn': 7,  # this project to cortex
                             'dcnp': 8,  # while this project to IO (there is dcnp_io connection) -> opposite to paper!!
                             'io': 9}

        self.hdf5_file_name = hdf5_file_name

        self.Cereb_pops, self.Cereb_pop_ids, self.WeightPFPC, self.PF_PC_conn = self.create_Cereb(nest, hdf5_file_name,mode, experiment, dopa_depl, LTD, LTP)
        
        background_pops = self.create_ctxinput(nest, pos_file=None, in_spikes='background')

        if not cortex_type:                                                                                          
            self.CTX_pops = background_pops
        else:
            self.CTX_pops = self.create_ctxinput(nest, pos_file=None, in_spikes=cortex_type, n_spike_generators=n_spike_generators)

    def create_Cereb(self, nest_, pos_file, mode, experiment, dopa_depl, LTD, LTP):
        ### Load neuron positions from hdf5 file and create them in NEST:
        with h5py.File(pos_file, 'r') as f:
            positions = np.array(f['positions'])

        if experiment == 'EBCC':
            plasticity = True
        else:
            plasticity = False

        id_2_cell_type = {val: key for key, val in self.cell_type_ID.items()}
        # Sort nrntype IDs
        sorted_nrn_types = sorted(list(self.cell_type_ID.values()))
        # Create a dictionary; keys = cell names, values = lists to store neuron models
        neuron_models = {key: [] for key in self.cell_type_ID.keys()}

        # All cells are modelled as E-GLIF models;
        # with the only exception of Glomeruli (not cells, just modeled as
        # relays; i.e., parrot neurons)
        for cell_id in sorted_nrn_types:
            cell_name = id_2_cell_type[cell_id]
            if cell_name != 'glomerulus':
                if cell_name not in nest_.Models():
                    nest_.CopyModel('eglif_cond_alpha_multisyn', cell_name)
                    nest_.SetDefaults(cell_name, neuron_param[cell_name])
            else:
                if cell_name not in nest_.Models():
                    nest_.CopyModel('parrot_neuron', cell_name)

            cell_pos = positions[positions[:, 1] == cell_id, :]
            n_cells = cell_pos.shape[0]
            neuron_models[cell_name] = nest_.Create(cell_name, n_cells)

            # initial value variation #TODO
            # if cell_name != 'glomerulus':
            #     dVinit = [{"Vinit": np.random.uniform(neuron_param[cell_name]['Vinit'] - 10,
            #                                           neuron_param[cell_name]['Vinit'] + 10)}
            #               for _ in range(n_cells)]

            #     nest_.SetStatus(neuron_models[cell_name], dVinit)

            # delete death PCs
            if cell_name == 'purkinje':
                if mode == 'internal_dopa' or mode == 'both_dopa':
                    n_PC_alive = int(cell_pos.shape[0] * (1. - 0.5 * (-dopa_depl) / 0.8))  # number of PC still alive
                else:
                    n_PC_alive = cell_pos.shape[0]

                all_purkinje = list(neuron_models['purkinje'])
                np.random.shuffle(all_purkinje)
                selected_purkinje = all_purkinje[:n_PC_alive]      # indexes of PC still alive
                death_purkinje = all_purkinje[n_PC_alive:]
                nest_.SetStatus(death_purkinje, neuron_param['death_purkinje'])


        with h5py.File(pos_file, 'r') as f:
            vt = {}
            for conn in conn_weights.keys():
                connection = np.array(f['connections/' + conn])
                pre = [int(x + 1) for x in connection[:, 0]]  # pre and post may contain repetitions!
                post = [int(x + 1) for x in connection[:, 1]]
                
                if "pf_pc" in conn:
                    if plasticity:
                        # Init_PFPC = np.random.uniform(conn_weights['pf_pc'] * 0.9, conn_weights['pf_pc'] * 1.1,
                        #                           size=len(pre[grc_selected_ids]))

                        # Create 1 volume transmitter for each post-synaptic neuron
                        vt[receiver[conn]] = nest_.Create("volume_transmitter_alberto",len(np.unique(post)))
                        print("Created vt for ", conn, " connections")
                        for n,vti in enumerate(vt[receiver[conn]]):
                            nest_.SetStatus([vti],{"vt_num" : n})
                        
                        # Set plastic connection parameters for stdp_synapse_sinexp synapse model
                        name_plast = 'plast_'+conn
                        nest_.CopyModel('stdp_synapse_sinexp', name_plast)
                        nest_.SetDefaults(name_plast,{"A_minus": LTD,   # double - Amplitude of weight change for depression
                                                    "A_plus": LTP,   # double - Amplitude of weight change for facilitation
                                                    "Wmin": 0.0,    # double - Minimum synaptic weight
                                                    "Wmax": 4000.0,     # double - Maximum synaptic weight
                                                    "vt": vt[receiver[conn]][0]})
                            
                        syn_param = {"model": name_plast, "weight": conn_weights[conn], "delay": conn_delays[conn], "receptor_type": conn_receptors[conn]}

                        # Create connection and associate a volume transmitter to them
                        for vt_num, post_cell in enumerate(np.unique(post)):
                                            syn_param["vt_num"] = float(vt_num)
                                            indexes = np.where(post == post_cell)[0]
                                            pre_neurons = np.array(pre)[indexes]
                                            post_neurons = np.array(post)[indexes]
                                            nest_.Connect(pre_neurons,post_neurons, {"rule": "one_to_one"}, syn_param)
                    else:

                        syn_param = {"model": "static_synapse", "weight": conn_weights[conn], "delay": conn_delays[conn],"receptor_type": conn_receptors[conn]}
                        nest_.Connect(pre,post, {"rule": "one_to_one"}, syn_param)
    
                # Static connections with distributed delay                                
                elif conn == "io_bc" or conn == "io_sc":
                    syn_param = {"model": "static_synapse", "weight": conn_weights[conn], \
                                "delay": {'distribution': 'normal_clipped', 'low': min_iomli, 'mu': conn_delays[conn],
                                'sigma': sd_iomli},"receptor_type":conn_receptors[conn]}
                    nest_.Connect(pre,post, {"rule": "one_to_one"}, syn_param)
                                                    
                # Static connections with constant delay
                else:
                    syn_param = {"model": "static_synapse", "weight": conn_weights[conn], "delay": conn_delays[conn],"receptor_type": conn_receptors[conn]}
                    nest_.Connect(pre,post, {"rule": "one_to_one"}, syn_param)
    
                # If a connection is a teaching one, also the corresponding volume transmitter should be connected
                if conn == "io_pc" and plasticity:                                     
                    post_n = np.array(post)-neuron_models[receiver[conn]][0] +vt[receiver[conn]][0]
                    nest_.Connect(np.asarray(pre, int), np.asarray(post_n, int), {"rule": "one_to_one"},{"model": "static_synapse", "weight": 1.0, "delay": 1.0})
        

                    print("Connections ", conn, " done!")
    
        Cereb_pops = neuron_models
        pop_ids = {key: (min(neuron_models[key]), max(neuron_models[key])) for key, _ in self.cell_type_ID.items()}
        WeightPFPC = None
        PF_PC_conn = None
        return Cereb_pops, pop_ids, WeightPFPC, PF_PC_conn


    def create_ctxinput(self, nest_, pos_file=None, in_spikes='poisson', n_spike_generators='n_glomeruli',
                        experiment='active', CS ={"start":500., "stop":760., "freq":36.}, US ={"start":750., "stop":760., "freq":500.}, tot_trials = None, len_trial = None):

        glom_id, _ = self.get_glom_indexes(self.Cereb_pops['glomerulus'], "EBCC")
        id_stim = sorted(list(set(glom_id)))
        n = len(id_stim)
        IO_id = self.Cereb_pops['io']

        if in_spikes == "background":
        # Background as Poisson process, always present

            CTX = nest_.Create('poisson_generator', len(self.Cereb_pops['glomerulus']),params={'rate': 4.0, 'start': 0.0})
            nest_.Connect(CTX, self.Cereb_pops['glomerulus'], {"rule":"one_to_one"})  # connected to all of them

        if in_spikes == 'spike_generator':
            print('The cortex input is a spike generator')

            if n_spike_generators == 'n_glomeruli':
                n_s_g = n  # create one spike generator for each input population
            else:
                n_s_g = n_spike_generators  # create n_s_g, randomly connected to the input population

            # create a cortex input
            CTX = nest_.Create("spike_generator", n_s_g)  # , params=generator_params)
            syn_param = {"delay": 2.0}

            # connect
            if n_spike_generators == 'n_glomeruli':
                nest_.Connect(CTX, id_stim, {'rule': 'one_to_one'}, syn_param)
            else:
                np.random.shuffle(id_stim)
                n_targets = len(id_stim) / n_s_g
                for i in range(n_s_g - 1):
                    post = id_stim[round(i * n_targets):round((i + 1) * n_targets)]
                    nest_.Connect([CTX[i]], post, {'rule': 'all_to_all'})
                post = id_stim[round((n_s_g - 1) * n_targets):]
                nest_.Connect([CTX[n_s_g - 1]], post, {'rule': 'all_to_all'}, syn_param)


        elif in_spikes == 'spike_generator_control': #simulated arm
            print('The cortex input is a spike generator')

            # create a cortex input
            id_stim, _ = self.get_glom_indexes(self.Cereb_pops['glomerulus'], experiment)
            CTX = nest_.Create("spike_generator", len(id_stim))  # , params=generator_params)
            syn_param = {"delay": 2.0}

            # connect
            nest_.Connect(CTX, id_stim, {'rule': 'one_to_one'}, syn_param)

            US_n = ()
            US_p = ()
            for ii in range(len(self.Cereb_pops['io'])):  # uncomment to have different IO input in microzones
                US_new = nest_.Create('spike_generator')
                if ii < len(self.Cereb_pops['io']) / 2:
                    US_n = US_n + US_new
                elif ii >= len(self.Cereb_pops['io']) / 2:
                    US_p = US_p + US_new

            # Connection to first half of IO, corresponding to first microzone
            syn_param = {"model": "static_synapse", "weight": 5.0, "delay": 0.1, "receptor_type": 1}
            nest_.Connect(US_n, self.Cereb_pops['io'][:int(len(self.Cereb_pops['io']) / 2)],
                          {'rule': 'one_to_one'}, syn_param)
            nest_.Connect(US_p, self.Cereb_pops['io'][int(len(self.Cereb_pops['io']) / 2):],
                          {'rule': 'one_to_one'}, syn_param)

            return {'CTX': CTX, 'US_n': US_n, 'US_p': US_p}


        elif in_spikes == 'dynamic_poisson':
            print('The cortex input is a poissonian process')

            CS_FREQ = 36.
            # Simulate a conscious stimulus
            CTX = nest_.Create('poisson_generator', params={'rate': CS_FREQ})
            nest_.Connect(CTX, id_stim)


        elif in_spikes == 'poisson': #EBCC
            print('The cortex input is a poissonian process')

            CS_START = CS["start"]   # beginning of stimulation
            CS_END = CS["stop"]     # end of stimulation
            CS_FREQ = CS["freq"]  # Frequency in Hz (considering the background at 4 Hz (sum of Poisson processes = Poisson proc with the sum of rates)

            # Simulate a conscious stimulus
            CTX = nest_.Create('poisson_generator', params={'rate': CS_FREQ, 'start': CS_START, 'stop': CS_END})
            nest_.Connect(CTX, id_stim)

            # US as burst
            US_START = US["start"] # beginning of stimulation
            US_END = US["stop"] # end of stimulation -> like CS_END!
            US_FREQ = US["start"]  # Frequency in Hzv

            spike_nums = np.int(np.round((US_FREQ * (US_END - US_START)) / 1000.))
            US_array = (np.round(np.linspace(US_START, US_END, spike_nums)))

            US = ()
            for ii in range(int(len(self.Cereb_pops['io']) / 2)):  # uncomment to have different IO input in microzones
                US_new = nest_.Create('spike_generator')
                nest_.SetStatus(US_new, {'spike_times': US_array})
                US = US + US_new
            RESOLUTION = nest_.GetKernelStatus("resolution")
            # Connection to first half of IO, corresponding to first microzone
            syn_param = {"model": "static_synapse", "weight": 55.0, "delay": RESOLUTION, "receptor_type": 1}
            nest_.Connect(US, self.Cereb_pops['io'][:int(len(self.Cereb_pops['io']) / 2)],
                          {'rule': 'one_to_one'}, syn_param)

        elif in_spikes == "EBCC":
            
            IO_id = self.Cereb_pops['io']
            glom_id, _ = self.get_glom_indexes(self.Cereb_pops['glomerulus'], "EBCC")
            id_stim = sorted(list(set(glom_id)))

            US_matrix = np.concatenate(
                            [
                                np.arange(US["start"], US["end"] + 2, 2)
                                + len_trial * t
                                for t in range(tot_trials)
                            ]
                        )
            
            US_stim = nest_.Create("spike_generator", len(IO_id), {"spike_times":US_matrix})
            
            nest_.Connect(US_stim, IO_id, "all_to_all", {"receptor_type": 1, "delay":1.,"weight":10.}) #10.

            self.US = US_stim

            n_mf = 24
            self.bins = int((CS["end"] - CS["start"])/n_mf)

            n_glom_x_mf = len(glom_id)/n_mf
            splits = [int(n_glom_x_mf)*i for i in range(1,n_mf+1)]
            glom_mf = np.split(np.asarray(glom_id),splits)
            self.map_glom = {}
            self.CS_stim = nest_.Create("spike_generator", n_mf)

            CS_matrix_start_pre = np.round((np.linspace(100.0, 228.0, 11)))
            CS_matrix_start_post = np.round((np.linspace(240.0, 368.0, 11)))

            CS_matrix_first_pre = np.concatenate([CS_matrix_start_pre + len_trial * t for t in range(tot_trials)])
            CS_matrix_first_post = np.concatenate([CS_matrix_start_post + len_trial * t for t in range(tot_trials)])
            
            CS_matrix = []

            for i in range(int(n_mf/2)):
                CS_matrix.append(CS_matrix_first_pre+i)
                CS_matrix.append(CS_matrix_first_post+i)
            
            for sg in range(len(self.CS_stim)):	
                    nest_.SetStatus(self.CS_stim[sg : sg + 1], params={"spike_times": CS_matrix[sg].tolist()})
                    nest_.Connect(self.CS_stim[sg : sg + 1], glom_mf[sg].tolist())
                
            CTX = self.CS_stim

        elif in_spikes == "EBCC2":
            n_mf = 24
            # glom_id, _ = self.get_glom_indexes(self.Cereb_pops['glomerulus'], "EBCC")
            # IO_id = self.Cereb_pops['io']
            n_glom_x_mf = len(glom_id)/n_mf
            splits = [int(n_glom_x_mf)*i for i in range(1,n_mf+1)]
            #splits[-1] +=resto
            glom_mf = np.split(np.asarray(glom_id),splits)

            CS_matrix_start_pre = np.round((np.linspace(100.0, 228.0, 11)))
            CS_matrix_start_post = np.round((np.linspace(240.0, 368.0, 11)))

            CS_matrix_first_pre = np.concatenate([CS_matrix_start_pre + len_trial * t for t in range(tot_trials)])
            CS_matrix_first_post = np.concatenate([CS_matrix_start_post + len_trial * t for t in range(tot_trials)])
            
            CS_matrix = []

            for i in range(int(n_mf/2)):
                CS_matrix.append(CS_matrix_first_pre+i)
                CS_matrix.append(CS_matrix_first_post+i)

            CS_stim = nest.Create("spike_generator", n_mf)
            for sg in range(len(CS_stim)):	
                nest.SetStatus(CS_stim[sg : sg + 1], params={"spike_times": CS_matrix[sg].tolist()})
                nest.Connect(CS_stim[sg : sg + 1], glom_mf[sg].tolist())
            

            US_matrix = np.concatenate(
                            [
                                np.arange(US["start"]+ set_time, set_time + US["stop"] + 2, 2)
                                + len_trial * t
                                for t in range(tot_trials)
                            ]
                        )
            
            US_stim = nest.Create("spike_generator", len(IO_id), {"spike_times":US_matrix})
            
            nest.Connect(US_stim, IO_id, "all_to_all", {"receptor_type": 1, "delay":1.,"weight":10.}) #10.

            CTX = CS_matrix
            
        elif in_spikes == "EBCC1":
            

            spike_nums_CS = np.int(np.round((CS["freq"] * (CS["stop"] - CS["start"])) / 1000.))         # Rancz
            CS_matrix_start = np.random.uniform(CS["start"],CS["stop"],[len(self.Cereb_pops['glomerulus']), spike_nums_CS])

            CS_matrix = CS_matrix_start

            for t in range(1,tot_trials):
                CS_matrix = np.concatenate((CS_matrix,CS_matrix_start+t*len_trial),axis=1)

            CS = []

            for gn in range(len(cereb.Cereb_pops['glomerulus'])):
                spk_gen = nest.Create('spike_generator', params = {'spike_times': np.sort(np.round(CS_matrix[gn,:]))})
                CS.append(spk_gen[0])

            nest.Connect(list(CS[:n]), id_stim, {'rule': 'one_to_one'})

            spike_nums = np.int(np.round((US["freq"] * (US["stop"] - US["start"])) / 1000.))
            US_array = []
            for t in range(tot_trials):
                US_array.extend(np.round(np.linspace(t*len_trial+US["start"], t*len_trial+US["stop"], spike_nums)))

            US = ()
            split = 2
            for ii in range(int(len(IO_id)/split)):
                US_new = nest.Create('spike_generator')
                nest.SetStatus(US_new, {'spike_times': US_array})
                US = US + US_new

            # Connection to first half of IO, corresponding to first microzone
            syn_param = {"model": "static_synapse", "weight":90.0, "delay": 1.,"receptor_type":1}
            nest.Connect(US,IO_id[:int(len(IO_id)/split)],{'rule':'one_to_one'},syn_param)

            CTX = CS_matrix

        else:
            print("ATTENTION! no cortex input generated")
            CTX = []
            pass

        return {'CTX': CTX}

    def get_glom_positions_xz(self):
        _, idx = self.get_glom_indexes(self.Cereb_pops['glomerulus'])
        with h5py.File(self.hdf5_file_name, 'r') as f:
            positions = np.array(f['positions'])
            gloms_pos = positions[positions[:, 1] == self.cell_type_ID['glomerulus'], :]
            gloms_pos_xz = gloms_pos[:, [2, 4]]
        return gloms_pos_xz[idx, :]

    def get_glom_indexes(self, glom_pop, experiment):
        with h5py.File(self.hdf5_file_name, 'r') as f:
            positions = np.array(f['positions'])
            glom_posi = positions[positions[:, 1] == self.cell_type_ID['glomerulus'], :]
            glom_xz = glom_posi[:, [2, 4]]

            if experiment == 'EBCC' or experiment == 'active':
                x_c, z_c = 200., 200.

                RADIUS = 150.  # [um] - radius of glomeruli stimulation cylinder to avoid border effects
                # Connection to glomeruli falling into the selected volume, i.e. a cylinder in the Granular layer
                bool_idx = np.sum((glom_xz - np.array([x_c, z_c])) ** 2, axis=1).__lt__(
                    RADIUS ** 2)  # lt is less then, <
                target_gloms = glom_posi[bool_idx, 0] + 1
                id_stim = list(set([glom for glom in glom_pop if glom in target_gloms]))

            elif experiment == 'robot':
                x_high_bool = np.array(glom_xz[:, 0].__gt__(200 - 150))      # (200 - 120))  # z > 200 (left in paper)
                x_low_bool = np.array(glom_xz[:, 0].__lt__(200 + 150))     # (200 + 120))  # z > 200 (left in paper)
                z_high_bool = np.array(glom_xz[:, 1].__gt__(200 - 150))       # (200 - 20))  # 180 < z < 220 (right in paper)
                z_low_bool = np.array(glom_xz[:, 1].__lt__(200 + 150))      # (200 + 20))
                bool_idx = x_low_bool & x_high_bool & z_low_bool & z_high_bool# 180 < z < 220 (right in paper)
                idx = glom_posi[bool_idx, 0] + 1
                id_stim = list(set([glom for glom in glom_pop if glom in idx]))

        return id_stim, bool_idx

    def get_glom_indexes_pos_vel(self):
        _, idx = self.get_glom_indexes(self.Cereb_pops['glomerulus'])
        with h5py.File(self.hdf5_file_name, 'r') as f:
            positions = np.array(f['positions'])
            glom_posi = positions[positions[:, 1] == self.cell_type_ID['glomerulus'], :]
            glom_xz = glom_posi[:, [2, 4]]
            glom_xz = glom_xz[idx, :]
            z_pos_bool = np.array(glom_xz[:, 1].__lt__(200))      # (200 + 20))
            z_vel_bool = np.array(glom_xz[:, 1].__gt__(200))       # (200 - 20))  # 180 < z < 220 (right in paper)
            position_gloms = glom_xz[z_pos_bool, :]
            velocity_gloms = glom_xz[z_vel_bool, :]
        return position_gloms, velocity_gloms, z_pos_bool, z_vel_bool

    def get_dcn_indexes(self):
        with h5py.File(self.hdf5_file_name, 'r') as f:
            positions = np.array(f['positions'])
            dcn_posi = positions[positions[:, 1] == self.cell_type_ID['dcn'], :]
            dcn_xz = dcn_posi[:, [2, 4]]
            pos_bool = dcn_xz[:, 1].__gt__(200)  # z > 200 (left in paper)
            neg_bool = dcn_xz[:, 1].__lt__(200)  # z < 200 (right in paper)
            pos_idx = dcn_posi[pos_bool, 0] + 1
            neg_idx = dcn_posi[neg_bool, 0] + 1
            id_stim_pos = sorted(list(set([dcn for dcn in self.Cereb_pops['dcn'] if dcn in pos_idx])))
            id_stim_neg = sorted(list(set([dcn for dcn in self.Cereb_pops['dcn'] if dcn in neg_idx])))
        return id_stim_pos, id_stim_neg


if __name__ == "__main__":
    pass 
    import sys
    sys.path.append('/home/modelling/Desktop/benedetta/BGs_Cereb_nest_PD/')
    import nest
    from pathlib import Path
    from marco_nest_utils import utils
    import pickle
    CORES = 24
    VIRTUAL_CORES = 24
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
    Cereb_recorded_names = ['glomerulus', 'purkinje', 'dcn','dcnp', 'io']
    
    CS ={"start":100., "stop":380., "freq":50.}
    US ={"start":350., "stop":380., "freq":500.}
    
    baseline = 200
    len_trial = CS["stop"] + baseline
    set_time = 0

    # ltd = np.logspace(-4,0,base=2,num=5)
    # ltp = np.logspace(-10,-6,base=2,num=5)

    ltp = [0.000014]
    ltd = [0.000004]
    i=0

    for cortex_type in ["EBCC2"]:
        for LTP in ltp:
            for LTD in ltd:
                i +=1

                nest.ResetKernel()
                cereb = Cereb_class(nest, hdf5_file_name, n_spike_generators='n_glomeruli',
                            mode='external_dopa', experiment='EBCC', dopa_depl=0, LTD=-LTD, LTP=LTP)
            
                ct = cereb.create_ctxinput(nest, pos_file=hdf5_file_name, in_spikes=cortex_type, 
                                    experiment='EBCC', CS =CS, US =US, tot_trials = tot_trials, len_trial = len_trial)
                recorded_list = [cereb.Cereb_pops[name] for name in Cereb_recorded_names]
                sd_list = utils.attach_spikedetector(nest, recorded_list)
                
                model_dict = utils.create_model_dictionary(0, Cereb_recorded_names, {**cereb.Cereb_pop_ids}, len_trial,
                                                            sample_time=1., settling_time=set_time,
                                                            trials=tot_trials, b_c_params=[])
                

                print("Simulating settling time: " + str(set_time) )

                #nest.Simulate(set_time)

                
                for trial in range(tot_trials):
                    
                    '''
                    # CS_spk = np.around(np.linspace(CS["start"]+ set_time +(trial*len_trial),CS["stop"]+ set_time +(trial*len_trial),22), decimals=1)
                    # CS_stim = nest.Create("spike_generator", len(glom_id), {"spike_times":CS_spk})

                    # CS_stim = nest.Create("poisson_generator", len(glom_id), {"start":500.+(trial*len_trial), "stop":760.+(trial*len_trial), "rate":36.})
                    #nest.Connect(CS_stim, glom_id, "one_to_one")
                    '''

                    '''
                    # US_spk = np.around(np.linspace(US["start"]+ set_time +(trial*len_trial),US["stop"]+ set_time +(trial*len_trial),int(US["freq"]*1000/(US["stop"]-US["start"]))), decimals=1)
                    # US_stim = nest.Create("spike_generator", len(IO_id), {"spike_times":US_spk})
                    
                    # US_stim = nest.Create("poisson_generator", len(IO_id), {"start":750.+(trial*len_trial), "stop":760.+(trial*len_trial), "rate":200.})
                    nest.Connect(US_stim, IO_id, "all_to_all", {"receptor_type": 1, "delay":1.,"weight":10.})
                    '''
                    
                    print("Simulating trial: " + str(trial +1) +" di "+ str(tot_trials))
                    print(LTP,LTD,cortex_type)
                    print(i)
                    nest.Simulate(len_trial)

                    
                rasters = utils.get_spike_values(nest, sd_list, Cereb_recorded_names)
                with open(f'./cereb_test/rasters_trials_'+cortex_type+"_"+str(tot_trials)+'_LTP_'+str(LTP)+"_LTD_"+str(LTD)+"_test", 'wb') as pickle_file:
                    pickle.dump(rasters, pickle_file)


                with open(f'./cereb_test/model_dict_trials_'+cortex_type+"_"+str(tot_trials)+'_LTP_'+str(LTP)+"_LTD_"+str(LTD)+"_test", 'wb') as pickle_file:
                    pickle.dump(model_dict, pickle_file)