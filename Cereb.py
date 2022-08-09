# -*- coding: utf-8 -*-
"""
"""

__author__ = 'marco'

import numpy as np
import h5py
from copy import deepcopy

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
neuron_param = {
    'golgi': {'t_ref': 2.0, 'C_m': 145.0, 'tau_m': 44.0, 'V_th': -55.0, 'V_reset': -75.0, 'Vinit': -62.0, 'E_L': -62.0,
              'Vmin': -150.0,
              'lambda_0': 1.0, 'tau_V': 0.4, 'I_e': 16.214, 'kadap': 0.217, 'k1': 0.031, 'k2': 0.023, 'A1': 259.988,
              'A2': 178.01,
              'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['golgi'],
              'tau_syn2': tau_inh['golgi'], 'tau_syn3': tau_exc_pfgoc},
    'granule': {'t_ref': 1.5, 'C_m': 7.0, 'tau_m': 24.15, 'V_th': -41.0, 'V_reset': -70.0, 'Vinit': -62.0, 'E_L': -62.0,
                'Vmin': -150.0,
                'lambda_0': 1.0, 'tau_V': 0.3, 'I_e': -0.888, 'kadap': 0.022, 'k1': 0.311, 'k2': 0.041, 'A1': 0.01,
                'A2': -0.94,
                'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['granule'],
                'tau_syn2': tau_inh['granule'], 'tau_syn3': tau_exc['granule']},
    'purkinje': {'t_ref': 0.5, 'C_m': 334.0, 'tau_m': 47.0, 'V_th': -43.0, 'V_reset': -69.0, 'Vinit': -59.0,
                 'E_L': -59.0,
                 'lambda_0': 4.0, 'tau_V': 3.5, 'I_e': 742.54, 'kadap': 1.492, 'k1': 0.1950, 'k2': 0.041, 'A1': 157.622,
                 'A2': 172.622,
                 'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['purkinje'],
                 'tau_syn2': tau_inh['purkinje'], 'tau_syn3': tau_exc_cfpc},
    'basket': {'t_ref': 1.59, 'C_m': 14.6, 'tau_m': 9.125, 'V_th': -53.0, 'V_reset': -78.0, 'Vinit': -68.0,
               'E_L': -68.0,
               'lambda_0': 1.8, 'tau_V': 1.1, 'I_e': 3.711, 'kadap': 2.025, 'k1': 1.887, 'k2': 1.096, 'A1': 5.953,
               'A2': 5.863,
               'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['basket'],
               'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
    'stellate': {'t_ref': 1.59, 'C_m': 14.6, 'tau_m': 9.125, 'V_th': -53.0, 'V_reset': -78.0, 'Vinit': -68.0,
                 'E_L': -68.0,
                 'lambda_0': 1.8, 'tau_V': 1.1, 'I_e': 3.711, 'kadap': 2.025, 'k1': 1.887, 'k2': 1.096, 'A1': 5.953,
                 'A2': 5.863,
                 'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['basket'],
                 'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
    'dcn': {'t_ref': 0.8, 'C_m': 142.0, 'tau_m': 33.0, 'V_th': -36.0, 'V_reset': -55.0, 'Vinit': -45.0, 'E_L': -45.0,
            'lambda_0': 3.5, 'tau_V': 3.0, 'I_e': 75.385, 'kadap': 0.408, 'k1': 0.697, 'k2': 0.047, 'A1': 13.857,
            'A2': 3.477,
            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['dcn'],
            'tau_syn2': tau_inh['dcn'], 'tau_syn3': tau_exc['dcn']},
    'dcnp': {'t_ref': 0.8, 'C_m': 56.0, 'tau_m': 56.0, 'V_th': -39.0, 'V_reset': -55.0, 'Vinit': -40.0, 'E_L': -40.0,
             'lambda_0': 0.9, 'tau_V': 1.0, 'I_e': 2.384, 'kadap': 0.079, 'k1': 0.041, 'k2': 0.044, 'A1': 176.358,
             'A2': 176.358,
             'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['dcnp'],
             'tau_syn2': tau_inh['dcnp'], 'tau_syn3': tau_exc['dcnp']},
    'io': {'t_ref': 1.0, 'C_m': 189.0, 'tau_m': 11.0, 'V_th': -35.0, 'V_reset': -45.0, 'Vinit': -45.0, 'E_L': -45.0,
           'lambda_0': 1.2, 'tau_V': 0.8, 'I_e': -18.101, 'kadap': 1.5, 'k1': 0.191, 'k2': 0.091, 'A1': 1810.93,
           'A2': 1358.197,  # 'kadap': 1.928
           'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['io'],
           'tau_syn2': tau_inh['io'], 'tau_syn3': tau_exc['io']}}

# Connection weights
conn_weights = {'aa_goc': 1.2, 'aa_pc': 0.7, 'bc_pc': 0.3, 'dcnp_io': 3.0, 'gj_bc': 0.2, 'gj_sc': 0.2, 'glom_dcn': 0.05, \
                'glom_goc': 1.5, 'glom_grc': 0.15, 'goc_glom': 0.0, 'gj_goc': 0.3, 'goc_grc': 0.6, 'io_dcn': 0.1,
                'io_dcnp': 0.2,
                'io_bc': 1.0, 'io_sc': 1.0, 'io_pc': 350.0, 'pc_dcn': 0.4, 'pc_dcnp': 0.12, 'pf_bc': 0.015,
                'pf_goc': 0.05, 'pf_pc': 0.007,
                'pf_sc': 0.015, 'sc_pc': 0.3}

# Connection delays
conn_delays = {'aa_goc': 2.0, 'aa_pc': 2.0, 'bc_pc': 4.0, 'dcnp_io': 20.0, 'gj_bc': 1.0, 'gj_sc': 1.0, 'glom_dcn': 4.0,
               'glom_goc': 4.0, 'glom_grc': 4.0, 'goc_glom': 0.5, 'gj_goc': 1.0, 'goc_grc': 2.0, 'io_dcn': 4.0,
               'io_dcnp': 5.0,
               'io_bc': 70.0, 'io_sc': 70.0, 'io_pc': 4.0, 'pc_dcn': 4.0, 'pc_dcnp': 4.0, 'pf_bc': 5.0, 'pf_goc': 5.0,
               'pf_pc': 5.0,
               'pf_sc': 5.0, 'sc_pc': 5.0}

sd_iomli = 10.0  # IO-MLI delayes are set as normal distribution to reproduce the effect of spillover-based transmission
min_iomli = 40.0

# Connection receptors
conn_receptors = {'aa_goc': 3, 'aa_pc': 1, 'bc_pc': 2, 'dcnp_io': 2, 'gj_bc': 2, 'gj_sc': 2, 'glom_dcn': 1,
                  'glom_goc': 1, 'glom_grc': 1, 'goc_glom': 1, 'gj_goc': 2, 'goc_grc': 2, 'io_dcn': 1, 'io_dcnp': 1,
                  'io_bc': 3, 'io_sc': 3, 'io_pc': 3, 'pc_dcn': 2, 'pc_dcnp': 2, 'pf_bc': 1, 'pf_goc': 3, 'pf_pc': 1,
                  'pf_sc': 1, 'sc_pc': 2}


class Cereb_class:
    def __init__(self, nest, hdf5_file_name, cortex_type, n_spike_generators='n_glomeruli',
                 mode='external_dopa', experiment='active', dopa_depl=0, LTD=None):
        # create Basal Ganglia neurons and connections
        # self.N = number_of_neurons  # total BGs pop neurons
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

        self.Cereb_pops, self.Cereb_pop_ids, self.WeightPFPC, self.PF_PC_conn = self.create_Cereb(nest, hdf5_file_name,
                                                                                                  mode, experiment, dopa_depl, LTD)
        # cortex type identifies the type of input given by the Cortex: poissonian or spike generator
        self.CTX_pops = self.create_ctxinput(nest, hdf5_file_name, in_spikes=cortex_type,
                                             n_spike_generators=n_spike_generators, mode=mode)

    def create_Cereb(self, nest_, pos_file, mode, experiment, dopa_depl, LTD):
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

            if cell_name == 'purkinje':
                if mode == 'internal_dopa':
                    n_PC_alive = int(cell_pos.shape[0] * (1. - 0.5 * (-dopa_depl) / 0.8))  # number of PC still alive
                else:
                    n_PC_alive = cell_pos.shape[0]

                selected_purkinje = list(neuron_models['purkinje'])
                np.random.shuffle(selected_purkinje)
                selected_purkinje = selected_purkinje[:n_PC_alive]      # indexes of PC still alive

            # initial value variation
            if cell_name != 'glomerulus':
                dVinit = [{"Vinit": np.random.uniform(neuron_param[cell_name]['Vinit'] - 10,
                                                      neuron_param[cell_name]['Vinit'] + 10)}
                          for _ in range(n_cells)]

                nest_.SetStatus(neuron_models[cell_name], dVinit)

        ### Load connections from hdf5 file and create them in NEST:
        with h5py.File(pos_file, 'r') as f:
            if plasticity:
                # create volume transmitters
                # if mode == 'internal_dopa':
                #     pc_num = n_PC_alive
                # else:
                pc_num = max(neuron_models['purkinje']) - min(neuron_models['purkinje']) + 1
                vt = nest_.Create("volume_transmitter_alberto", pc_num)
                for n, vti in enumerate(vt):
                    nest_.SetStatus([vti], {"vt_num": n})

                # # connect weight recorder
                # pf = neuron_models['granule']  # here all pf are present once
                # pc = neuron_models['purkinje'][:pc_num]
                # # here all pc are present once
                # pf_idx = [i for i, p in enumerate(pf) if p in [7714, 19132]]
                # pc_idx = [i for i, p in enumerate(pc) if p in [95514, 95473]]
                # recdict = {"to_memory": True,
                #            "to_file": False,
                #            "label": "PFPC_",
                #            "senders": [pf[i] for i in pf_idx],
                #            "targets": [pc[i] for i in pc_idx]}
                # WeightPFPC = nest_.Create('weight_recorder', params=recdict)
                WeightPFPC = None

                # create pf_pc connection. To be done before io_pc
                connection = np.array(f['connections/pf_pc'])
                pre = np.array([int(x + 1) for x in connection[:, 0]])      # PF  # pre and post may contain repetitions!
                post = np.array([int(x + 1) for x in connection[:, 1]])     # PC

                if LTD is not None:
                    LTD1 = LTD # -1.0e-3*2  # -1.0e-3      # 1/10. than Antonietti test since weight is 1/10.
                else:
                    LTD1 = -1.0e-3*2
                LTP1 = 1.0e-4*0.5
                nest_.SetDefaults('stdp_synapse_sinexp',
                                  {"A_minus": LTD1,  # -1.0e-2
                                   "A_plus": LTP1,  # 1.0e-3
                                   "Wmin": 0.0,
                                   "Wmax": conn_weights['pf_pc'] * 10,  # 4.0
                                   "vt": vt[0]})
                                   # "weight_recorder": WeightPFPC[0]})

                select_plasticity = True
                if select_plasticity:
                    # define plasticity only on PC receiving from granule cells connected to MF!
                    # (only central circle is connected)
                    connection2 = np.array(f['connections/glom_grc'])
                    pre_glom = np.array([int(x + 1) for x in connection2[:, 0]])
                    post_grc = np.array([int(x + 1) for x in connection2[:, 1]])

                    plastic_ids, _ = self.get_glom_indexes(neuron_models['glomerulus'], experiment)
                    # these are the glom cells receving from input:
                    glom_selected_ids = np.isin(pre_glom, plastic_ids)
                    grc_ids = np.unique(post_grc[glom_selected_ids])
                    grc_selected_ids = np.isin(pre, grc_ids)

                    # for now, connect the ones without plasticity
                    syn_param = {"model": "static_synapse", "weight": conn_weights['pf_pc'],
                                 "delay": conn_delays['pf_pc'],
                                 "receptor_type": conn_receptors['pf_pc']}
                    nest_.Connect(pre[np.logical_not(grc_selected_ids)], post[np.logical_not(grc_selected_ids)],
                                  {'rule': 'one_to_one',
                                   "multapses": False},
                                  syn_param)
                else:
                    grc_selected_ids = np.array(np.ones(len(pre)), dtype=bool)

                # PF-PC excitatory plastic connections
                post_array = np.array(post, int)
                if mode == 'internal_dopa':
                    # extract only the PC still alive
                    pc_selected_ids = np.isin(post, selected_purkinje)
                    grc_selected_ids = np.logical_and(grc_selected_ids, pc_selected_ids)

                # Init_PFPC = conn_weights['pf_pc']
                Init_PFPC = np.random.uniform(conn_weights['pf_pc'] * 0.9, conn_weights['pf_pc'] * 1.1,
                                              size=len(pre[grc_selected_ids]))

                idx = np.array((post_array - post_array.min()).tolist())  # list of vt_num, one for each connection
                syn_param = {"model": 'stdp_synapse_sinexp',
                             "weight": Init_PFPC,
                             "delay": conn_delays['pf_pc'],
                             "receptor_type": conn_receptors['pf_pc'],
                             "vt_num": idx[grc_selected_ids], }
                nest_.Connect(pre[grc_selected_ids], post[grc_selected_ids],
                              {'rule': 'one_to_one',
                               "multapses": False},
                              syn_param)
                PF_PC_conn = nest_.GetConnections(neuron_models['granule'], neuron_models['purkinje'])

                print("Connections ", 'pf_pc', " done!")
            else:
                WeightPFPC = None
                PF_PC_conn = None

            for conn in conn_weights.keys():
                # exec(conn + " = np.array(f['connections/'+conn])")
                connection = np.array(f['connections/' + conn])
                # exec("pre = [int(x+1) for x in " + conn + "[:,0]]")
                pre = [int(x + 1) for x in connection[:, 0]]  # pre and post may contain repetitions!
                # exec("post = [int(x+1) for x in " + conn + "[:,1]]")
                post = [int(x + 1) for x in connection[:, 1]]

                if not plasticity:
                    if conn == "io_bc" or conn == "io_sc":
                        syn_param = {"model": "static_synapse", "weight": conn_weights[conn],
                                     "delay": {'distribution': 'normal_clipped', 'low': min_iomli,
                                               'mu': conn_delays[conn],
                                               'sigma': sd_iomli}, "receptor_type": conn_receptors[conn]}
                    else:
                        syn_param = {"model": "static_synapse", "weight": conn_weights[conn],
                                     "delay": conn_delays[conn],
                                     "receptor_type": conn_receptors[conn]}

                    nest_.Connect(pre, post, {"rule": "one_to_one"}, syn_param)
                    print("Connections ", conn, " done!")

                else:  # if plasticity
                    if conn not in ["pf_pc", 'io_pc']:
                        ### every other connection ###
                        # if conn not in ['aa_pc', 'bc_pc', 'sc_pc', 'io_sc', 'io_bc', 'io_dcnp', 'io_dcn', 'dcnp_io']:
                        if conn == "io_bc" or conn == "io_sc":
                            syn_param = {"model": "static_synapse", "weight": conn_weights[conn],
                                         "delay": {'distribution': 'normal_clipped', 'low': min_iomli,
                                                   'mu': conn_delays[conn],
                                                   'sigma': sd_iomli}, "receptor_type": conn_receptors[conn]}
                            nest_.Connect(pre, post, {"rule": "one_to_one"}, syn_param)
                        elif conn in ['aa_pc', 'bc_pc', 'sc_pc']:
                            # connect only the PC still alive
                            pc_selected_ids = np.isin(post, selected_purkinje)
                            syn_param = {"model": "static_synapse", "weight": conn_weights[conn],
                                         "delay": {'distribution': 'normal_clipped', 'low': min_iomli,
                                                   'mu': conn_delays[conn],
                                                   'sigma': sd_iomli}, "receptor_type": conn_receptors[conn]}
                            nest_.Connect(np.array(pre, int)[pc_selected_ids], np.array(post, int)[pc_selected_ids], {"rule": "one_to_one"}, syn_param)
                        elif conn in ['pc_dcn', 'pc_dcnp']:
                            # connect only the PC still alive
                            pc_selected_ids = np.isin(pre, selected_purkinje)
                            syn_param = {"model": "static_synapse", "weight": conn_weights[conn],
                                         "delay": {'distribution': 'normal_clipped', 'low': min_iomli,
                                                   'mu': conn_delays[conn],
                                                   'sigma': sd_iomli}, "receptor_type": conn_receptors[conn]}
                            nest_.Connect(np.array(pre, int)[pc_selected_ids], np.array(post, int)[pc_selected_ids], {"rule": "one_to_one"}, syn_param)
                        else:
                            syn_param = {"model": "static_synapse", "weight": conn_weights[conn],
                                         "delay": conn_delays[conn],
                                         "receptor_type": conn_receptors[conn]}
                            nest_.Connect(pre, post, {"rule": "one_to_one"}, syn_param)

                        print("Connections ", conn, " done!")

                    elif conn == "io_pc":
                        ### connection io - pc ###
                        # ! io_pc substituted by io_vt
                        idx = [p - min(post) for p in post]  # get pc order, from 0 to n_pc
                        vt = [vt[i] for i in idx]  # reorder vt according to pc new order (before were both ascending)
                        pc_selected_ids = np.isin(post, selected_purkinje)
                        nest_.Connect(np.array(pre, int)[pc_selected_ids], np.array(vt, int)[pc_selected_ids],
                                      {'rule': 'one_to_one'},  # connected one to one
                                      {"model": "static_synapse",  # "receptor_type": conn_receptors[conn]
                                       "weight": 1.0,  # conn_weights[conn]
                                       "delay": conn_delays[conn]})  # 1.0

                        print("Connections  io_vt  done!")

                    # elif conn != "pf_pc" already done before

        Cereb_pops = neuron_models
        pop_ids = {key: (min(neuron_models[key]), max(neuron_models[key])) for key, _ in self.cell_type_ID.items()}
        return Cereb_pops, pop_ids, WeightPFPC, PF_PC_conn


    def create_ctxinput(self, nest_, pos_file=None, in_spikes='poisson', n_spike_generators='n_glomeruli',
                        mode='external_dopa', experiment='active'):
        # position glomeruli
        with h5py.File(pos_file, 'r') as f:
            positions = np.array(f['positions'])
        gloms_pos = positions[positions[:, 1] == self.cell_type_ID['glomerulus'], :]
        x_c, z_c = 200., 200.

        RADIUS = 150.  # [um] - radius of glomeruli stimulation cylinder to avoid border effects
        # Connection to glomeruli falling into the selected volume, i.e. a cylinder in the Granular layer
        target_gloms_idx = np.sum((gloms_pos[:, [2, 4]] - np.array([x_c, z_c])) ** 2, axis=1).__lt__(
            RADIUS ** 2)  # lt is less then, <
        target_gloms = gloms_pos[target_gloms_idx, 0] + 1
        id_stim = [glom for glom in self.Cereb_pops['glomerulus'] if glom in target_gloms]
        id_stim = sorted(list(set(id_stim)))
        n = len(id_stim)

        # Background as Poisson process, always present
        BG_CTX = nest_.Create('poisson_generator', params={'rate': 4.0, 'start': 0.0})
        nest_.Connect(BG_CTX, self.Cereb_pops['glomerulus'])  # connected to all of them

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


        elif in_spikes == 'spike_generator_control':
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


        elif in_spikes == 'poisson':
            print('The cortex input is a poissonian process')

            CS_START = 500.   # beginning of stimulation
            CS_END = 760.     # end of stimulation
            CS_FREQ = 36.  # Frequency in Hz (considering the background at 4 Hz (sum of Poisson processes = Poisson proc with the sum of rates)

            # Simulate a conscious stimulus
            CTX = nest_.Create('poisson_generator', params={'rate': CS_FREQ, 'start': CS_START, 'stop': CS_END})
            nest_.Connect(CTX, id_stim)

            # US as burst
            US_START = 750 # beginning of stimulation
            US_END = 760 # end of stimulation -> like CS_END!
            US_FREQ = 500.  # Frequency in Hzv

            spike_nums = np.int(np.round((US_FREQ * (US_END - US_START)) / 1000.))
            US_array = (np.round(np.linspace(US_START, US_END, spike_nums)))

            US = ()
            for ii in range(int(len(self.Cereb_pops['io']) / 2)):  # uncomment to have different IO input in microzones
                US_new = nest_.Create('spike_generator')
                nest_.SetStatus(US_new, {'spike_times': US_array})
                US = US + US_new

            # Connection to first half of IO, corresponding to first microzone
            syn_param = {"model": "static_synapse", "weight": 55.0, "delay": 0.1, "receptor_type": 1}
            nest_.Connect(US, self.Cereb_pops['io'][:int(len(self.Cereb_pops['io']) / 2)],
                          {'rule': 'one_to_one'}, syn_param)

        # in addition, if in conditioning scenario

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

            if experiment == 'EBCC':
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
