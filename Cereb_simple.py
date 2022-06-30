# -*- coding: utf-8 -*-
"""
"""

__author__ = 'marco'

import numpy as np

RECORD_WEIGHTS = False

# CEREBELLUM
PLAST1 = True  # PF-PC ex
PLAST2 = False  # MF-DCN ex
PLAST3 = False  # PC-DCN in

LTP1 = 0.1
LTD1 = -1.0

LTP2 = 1e-5
LTD2 = -1e-6
LTP3 = 1e-7
LTD3 = 1e-6

PROP_COEFF = 2.5

# Init_PFPC = 1.0
Init_PFPC = {'distribution': 'uniform',  # initial value of GR-PC synapse
             'low': 1.0 * 0.9, 'high': 1.0 * 1.1}

Init_MFDCN = 0.4 / PROP_COEFF  # 0.3 troppo poco, 0.5 troppo? ... was 0.4

Init_PCDCN = -1.0 * 3.

Init_MFGR_low = 1.0
Init_MFGR_high = 2.0


class Cereb_class:
    def __init__(self, nest, number_of_neurons, cortex_type, LTP_=LTP1, LTD_=LTD1, n_joints_=1):
        # create Basal Ganglia neurons and connections
        self.N = number_of_neurons  # total BGs pop neurons
        self.Cereb_pops, self.Cereb_pop_ids, self.WeightPFPC = self.create_Cereb(nest, LTP_, LTD_)

        # cortex type identifies the type of input given by the Cortex: poissonian or spike generator
        self.CTX_pops = self.create_ctxinput(nest, in_spikes_=cortex_type, n_joints_=n_joints_)

    def create_Cereb(self, nest_, LTP_, LTD_):
        nest_.CopyModel('iaf_cond_exp', 'granular_neuron')
        nest_.CopyModel('iaf_cond_exp', 'purkinje_neuron')
        nest_.CopyModel('iaf_cond_exp', 'olivary_neuron')
        nest_.CopyModel('iaf_cond_exp', 'nuclear_neuron')

        nest_.SetDefaults('granular_neuron', {'t_ref': 1.0,
                                              'C_m': 2.0,
                                              'V_th': -40.0,
                                              'V_reset': -70.0,
                                              'g_L': 0.2,
                                              'tau_syn_ex': 0.5,
                                              'tau_syn_in': 10.0})

        nest_.SetDefaults('purkinje_neuron', {'t_ref': 2.0,
                                              'C_m': 400.0,
                                              'V_th': -52.0,
                                              'V_reset': -70.0,
                                              'g_L': 16.0,
                                              'tau_syn_ex': 0.5,
                                              'tau_syn_in': 1.6})

        nest_.SetDefaults('olivary_neuron', {'t_ref': 1.0,
                                             'C_m': 2.0,
                                             'V_th': -40.0,
                                             'V_reset': -70.0,
                                             'g_L': 0.2,
                                             'tau_syn_ex': 0.5,
                                             'tau_syn_in': 10.0})

        nest_.SetDefaults('nuclear_neuron', {'t_ref': 1.0,
                                             'C_m': 2.0,
                                             'V_th': -40.0,
                                             'V_reset': -70.0,
                                             'g_L': 0.2,
                                             'tau_syn_ex': 0.5,
                                             'tau_syn_in': 10.0})

        # Cell numbers, according to Casellato et al., 2014
        MF_num = 300
        GR_num = MF_num * 20
        PC_num = 72
        IO_num = PC_num
        DCN_num = int(PC_num / 2)

        MF = nest_.Create("parrot_neuron", MF_num)
        GR = nest_.Create("granular_neuron", GR_num)
        PC = nest_.Create("purkinje_neuron", PC_num)
        IO = nest_.Create("olivary_neuron", IO_num)
        DCN = nest_.Create("nuclear_neuron", DCN_num)

        if PLAST1:
            vt = nest_.Create("volume_transmitter_alberto", PC_num)
            for n, vti in enumerate(vt):
                nest_.SetStatus([vti], {"vt_num": n})
        if PLAST2:
            vt2 = nest_.Create("volume_transmitter_alberto", DCN_num)
            for n, vti in enumerate(vt2):
                nest_.SetStatus([vti], {"vt_num": n})

        recdict2 = {"to_memory": RECORD_WEIGHTS,
                    "to_file": False,
                    "label": "PFPC_",
                    "senders": GR,
                    "targets": PC}

        WeightPFPC = nest_.Create('weight_recorder', params=recdict2)

        if PLAST3:
            nest_.SetDefaults('stdp_synapse', {"tau_plus": 30.0,
                                               "lambda": LTP3,
                                               "alpha": LTD3 / LTP3,
                                               "mu_plus": 0.0,  # Additive STDP
                                               "mu_minus": 0.0,  # Additive STDP
                                               "Wmax": -0.5,
                                               "weight": Init_PCDCN,
                                               "delay": 1.0})
            PCDCN_conn_param = {"model": "stdp_synapse"}
        else:
            PCDCN_conn_param = {"model": "static_synapse",
                                "weight": Init_PCDCN,
                                "delay": 1.0}

        MFGR_conn_param = {"model": "static_synapse",
                           "weight": {'distribution': 'uniform',
                                      # -> 0.75 GR fire at 7 Hz
                                      'low': Init_MFGR_low, 'high': Init_MFGR_high},  # fixed, not updated
                           "delay": 1.0}

        # MF-GR excitatory fixed connections
        # each GR receives 4 connections from 4 random moffy fibers
        nest_.Connect(MF, GR, {'rule': 'fixed_indegree',
                               'indegree': 4,
                               "multapses": False}, MFGR_conn_param)

        # A_minus - Amplitude of weight change for depression
        # A_plus - Amplitude of weight change for facilitation
        # Wmin - Minimal synaptic weight
        # Wmax - Maximal synaptic weight
        if PLAST1:
            nest_.SetDefaults('stdp_synapse_sinexp',
                              {"A_minus": LTD_,
                               "A_plus": LTP_,
                               "Wmin": 0.0,
                               "Wmax": 4.0,
                               "vt": vt[0],
                               "weight_recorder": WeightPFPC[0]})

            # PF-PC excitatory plastic connections
            # each PC receives the random 80% of the GR
            for i, PCi in enumerate(PC):
                PFPC_conn_param = {"model": 'stdp_synapse_sinexp',
                                   "weight": Init_PFPC,
                                   "delay": 1.0,
                                   'vt_num': i}

                nest_.Connect(GR, [PCi],
                              {'rule': 'fixed_indegree',
                               'indegree': int(0.8 * GR_num),
                               "multapses": False},
                              PFPC_conn_param)

        if PLAST1:
            # IO-PC teaching connections
            # Each IO is one-to-one connected with each PC
            nest_.Connect(IO, vt, {'rule': 'one_to_one'},  # connected one to one
                          {"model": "static_synapse",
                           "weight": 1.0, "delay": 1.0})
            IOPC_conn = nest_.GetConnections(IO, vt)  # IOPC_conn

        if PLAST2:
            # MF-DCN excitatory plastic connections
            # every MF is connected with every DCN
            nest_.SetDefaults('stdp_synapse_cosexp',
                              {"A_minus": LTD2,
                               "A_plus": LTP2,
                               "Wmin": 0.0,
                               "Wmax": 0.25,
                               "vt": vt2[0]})

            for i, DCNi in enumerate(DCN):
                MFDCN_conn_param = {"model": 'stdp_synapse_cosexp',
                                    "weight": Init_MFDCN,
                                    "delay": 10.0,
                                    'vt_num': i, }
                nest_.Connect(MF, [DCNi], 'all_to_all', MFDCN_conn_param)
                # A = nest_.GetConnections(MF, [DCNi])
                # nest_.SetStatus(A, {'vt_num': i})
        else:  # if not PLAST2
            MFDCN_conn_param = {"model": "static_synapse",
                                "weight": Init_MFDCN,
                                "delay": 10.0}
            nest_.Connect(MF, DCN, 'all_to_all', MFDCN_conn_param)

        # PC-DCN inhibitory plastic connections
        # each DCN receives 2 connections from 2 contiguous PC
        count_DCN = 0
        for P in range(PC_num):
            nest_.Connect([PC[P]], [DCN[count_DCN]],
                          'one_to_one', PCDCN_conn_param)
            if PLAST2:
                nest_.Connect([PC[P]], [vt2[count_DCN]], 'one_to_one',  # rivedi
                              {"model": "static_synapse",
                               "weight": 1.0,
                               "delay": 1.0})
            if P % 2 == 1:
                count_DCN += 1

        # PCDCN_conn = nest_.GetConnections(PC, DCN)  # PCDCN_conn
        Cereb_pops = {'MF': MF, 'GR': GR, 'PC': PC, 'IO': IO, 'DCN': DCN}
        pop_ids = {'MF': (min(MF), max(MF)), 'GR': (min(GR), max(GR)), 'PC': (min(PC), max(PC)),
                   'IO': (min(IO), max(IO)), 'DCN': (min(DCN), max(DCN))}
        return Cereb_pops, pop_ids, WeightPFPC

    def create_ctxinput(self, nest_, in_spikes_, n_joints_):
        # create MF input
        if in_spikes_ == 'spike_generator':
            spike_times = [10.]  # dummy value
            MF_num = self.Cereb_pop_ids['MF'][1] - self.Cereb_pop_ids['MF'][0] + 1
            generator_params = [{"spike_times": spike_times, "spike_weights": [1.] * len(spike_times)} for _ in
                                range(MF_num)]
            CTX = nest_.Create("spike_generator", MF_num, params=generator_params)
            # connect
            nest_.Connect(CTX, self.Cereb_pops['MF'], {'rule': 'one_to_one'})

            # print('!!! poisson background !!!')
            # TODO: try other background firing rates values
            generator_params = {"rate": 4.0}  # Antonietti 5.0
            CTX_bkg = nest_.Create("poisson_generator", params=generator_params)
            nest_.Connect(CTX_bkg, self.Cereb_pops['MF'],
                          {'rule': 'all_to_all'})  # replicate the same poisson train to all the targets

            US_n_list = []
            US_p_list = []
            sub_pop_IO_len = int(len(self.Cereb_pops['IO']) / n_joints_)

            for j in range(n_joints_):
                US_n = ()
                US_p = ()
                for ii in range(sub_pop_IO_len):  # uncomment to have different IO input in microzones
                    US_new = nest_.Create('spike_generator')
                    if ii < sub_pop_IO_len//2:
                        US_n = US_n + US_new
                    elif ii >= sub_pop_IO_len//2:
                        US_p = US_p + US_new

                assert len(US_n) == len(US_p), '+ and - poisson gen should have the same length'

                # Connection to first half of IO, corresponding to first microzone
                # TODO: try other weight values
                syn_param = {"model": "static_synapse", "weight": 10.0, "delay": 0.1}
                nest_.Connect(US_n, self.Cereb_pops['IO'][2*j*sub_pop_IO_len // 2:(2*j+1)*sub_pop_IO_len // 2],
                              {'rule': 'one_to_one'}, syn_param)
                nest_.Connect(US_p, self.Cereb_pops['IO'][(2*j+1)*sub_pop_IO_len // 2:(2*j+2)*sub_pop_IO_len // 2],
                              {'rule': 'one_to_one'}, syn_param)

                US_n_list += [US_n]
                US_p_list += [US_p]

        elif in_spikes_ == 'poisson':
            print('!!! poisson !!!')
            generator_params = {"rate": 11.0}
            CTX = nest_.Create("poisson_generator", params=generator_params)
            # generator_params = {"rate_times": [0.1], "rate_values": [5.0]}
            # CTX = nest_.Create("inhomogeneous_poisson_generator", params=generator_params)
            # connect
            nest_.Connect(CTX, self.Cereb_pops['MF'],
                          {'rule': 'all_to_all'})  # replicate the same poisson train to all the targets

            US_n_list = []
            US_p_list = []

        return {'CTX': CTX, 'US_p': US_p_list, 'US_n': US_n_list}


    def get_dcn_indexes(self, n_joints_):
        dim_DCN = len(self.Cereb_pops['DCN'])
        dim_DCN_joint = int(dim_DCN/(n_joints_*2))

        DCN_pop_list = []
        for j in range(n_joints_):
            DCN_pop_neg = [[self.Cereb_pops['DCN'][i] for i in range(2*j*dim_DCN_joint, (2*j+1)*dim_DCN_joint)]]  # 1st half
            DCN_pop_pos = [[self.Cereb_pops['DCN'][i] for i in range((2*j+1)*dim_DCN_joint, (2*j+2)*dim_DCN_joint)]]  # 2nd half
            DCN_pop_list += DCN_pop_neg
            DCN_pop_list += DCN_pop_pos

            assert len(DCN_pop_neg) == len(DCN_pop_pos), 'DCN + and - should have the same dimension'

        return DCN_pop_list
