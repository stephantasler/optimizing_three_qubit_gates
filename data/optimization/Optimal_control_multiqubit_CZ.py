import numpy as np
import matplotlib.pyplot as plt
import math
from qutip import *
import time
from scipy.optimize import minimize
import warnings
import pandas as pd
import os as os
import json
from mpl_toolkits.mplot3d import axes3d
import vector_generation as vg


import sys
sys.path.insert(0, './')
import helper as hlp
import time_dependence_correction as tdc


warnings.filterwarnings('ignore')

path = './Plots/'
path_temp = './Plots/Optimization_temp/'

timestr = time.strftime("%Y_%m_%d-%H:%M:%S")

######################################################################################## Plotting setup ########################################################################################

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)

np.set_printoptions(threshold=sys.maxsize)

##################################################################################### Defining circuit setup #####################################################################################

qubits_list = ['1','2','3','c1','c2']                 #qubits in the circuit
num_qubits = len(qubits_list)                         #number of qubits
num_data_qubits = 3
N = len(qubits_list) - 3                              #number of qubits, initialized in state 1 or 0, others are initialized in state 0
energy_level = [4,4,4,4,4]                    #number of fock states for each quantum object
energy_level_max = 4                                   #largest number of fock states for quantum object
energy_level_coupler = 4                              #number of fock states coupler
xyz_list = ['sx', 'sy', 'sz', 'splus', 'sminus']      #pauli-matrices
cr_an_list = ['b']                                    #ladder operator name
single_phase_qubits = 3

zero_pauli_strings = ['iix', 'iiy', 'ixi', 'ixz', 'iyi', 'iyz', 'izx', 'izy', 'xii', 'xiz', 'xxx', 'xxy', 'xyx', 'xyy', 'xzi', 'xzz', 'yii', 'yiz', 'yxx', 'yxy', 'yyx', 'yyy', 'yzi', 'yzz', 'zix', 'ziy', 'zxi', 'zxz', 'zyi', 'zyz', 'zzx', 'zzy']

############################################################################### Allocate operator and input vectors ###############################################################################

#allocate creation and anhilation operators
ladder_dict = {}
ladder_dict = hlp.get_creation_annihilation(qubits_list, cr_an_list, energy_level)
globals().update(ladder_dict)


# pre-allocate all states
psi_dict, psi_array, psi_bitstring, bitstring = hlp.get_vector(num_qubits,'01',energy_level,energy_level_max,qubits_list,ladder_dict,cr_an_list)
globals().update(psi_dict)


############################################################################### Import fault-tolerance breaking errors ###############################################################################

with open('../ft_breaking_errors/pauli_marginal_calculation/pauli_marginals.json', 'r') as f:
        ft_errors = json.load(f)

print(ft_errors)

######################################################################## Set couplings, anharmonicities and frequencies ########################################################################

pauli_op_dict = hlp.get_pauli_op_dict(num_qubits)


## frequencies ##

w1 = 4.89e9
w2 = 5.31e9
w3 = 4.83e9

wc1 = 7.496e9
wc2 = 7.44e9



## capacitances ##

C1 = 77.8e-15
C2 = 77.8e-15
C3 = 77.8e-15

Cc1 = 60.4e-15
Cc2 = 60.4e-15

C12 = 0.46e-15
C23 = 0.46e-15

C1c1 = 6.4e-15
C2c1 = 6.4e-15

C2c2 = 6.4e-15 
C3c2 = 6.4e-15 




######################################################################### Frequency, anharmonicity and coupling calculation #########################################################################

w_dict = {'w1': w1, 'wc1': wc1, 'w2': w2, 'wc2': wc2, 'w3': w3}
nodes = ['1', 'c1', '2', 'c2', '3']
node_mapping = {'1': 0, 'c1': 1, '2': 2, 'c2': 3, '3': 4}


C_mat = np.matrix([[C1c1+C12+C1,-C1c1,-C12,0,0],
                [-C1c1,C1c1+C2c1+Cc1,-C2c1,0,0],
                [-C12,-C2c1,C12+C2c1+C2c2+C23+C2,-C2c2,-C23],
                [0,0,-C2c2,C2c2+C3c2+Cc2,-C3c2],
                [0,0,-C23,-C3c2,C23+C3c2+C3]])



sys_params = hlp.get_coupling(frequency='omega', C_dict= 0, w_dict=w_dict, C_mat=C_mat, nodes=nodes)


H0 = hlp.get_H(params=sys_params, ladder_dict=ladder_dict)


comp_eigenstates_lables = vg.get_bitstrings(num_data_qubits,'01')

psi_000 = psi_00000
psi_001 = psi_00100
psi_010 = psi_01000
psi_011 = psi_01100

psi_100 = psi_10000
psi_101 = psi_10100
psi_110 = psi_11000
psi_111 = psi_11100


comp_eigenstates = [psi_000,psi_001,psi_010,psi_011,psi_100,psi_101,psi_110,psi_111]


evec_list, E_list, overlap_list, evec_dict, eval_dict, evec_label = hlp.get_eigenstate(H = H0, psi_array_temp = comp_eigenstates, psi_array_label = comp_eigenstates_lables, energy_level = energy_level, Multi = True)






evec_projectors = []

for i in range(len(comp_eigenstates)):
    evec_projectors.append(evec_list[i]*evec_list[i].dag())

################################################################################ Phase Gate ################################################################################

Pulse_c1 = {'Amp' : 0, 'slope' : 0, 't_gate': 0, 'Erfc': 'Erfc'}               # 3 parameter optimization
Pulse_c2 = {'Amp' : 0, 'slope' : 0, 't_gate': 0, 'Erfc': 'Erfc'}


sys_params['Pulse_c1'] = Pulse_c1
sys_params['Pulse_c2'] = Pulse_c2





################################################################# 50 ns Gate #################################################################

H = [H0,
    [ladder_dict['bc1'].dag()*ladder_dict['bc1'], tdc.Flux_pulse_c1], 
    [ladder_dict['bc2'].dag()*ladder_dict['bc2'], tdc.Flux_pulse_c2], 
    [(ladder_dict['b1'].dag() - ladder_dict['b1'])*(ladder_dict['bc1'].dag() - ladder_dict['bc1']), tdc.coupling_correction_1c1], 
    [(ladder_dict['b3'].dag() - ladder_dict['b3'])*(ladder_dict['bc1'].dag() - ladder_dict['bc1']), tdc.coupling_correction_c13],
    [(ladder_dict['b1'].dag() - ladder_dict['b1'])*(ladder_dict['bc2'].dag() - ladder_dict['bc2']), tdc.coupling_correction_1c2],
    [(ladder_dict['b3'].dag() - ladder_dict['b3'])*(ladder_dict['bc2'].dag() - ladder_dict['bc2']), tdc.coupling_correction_c23],
    [(ladder_dict['b2'].dag() - ladder_dict['b2'])*(ladder_dict['bc1'].dag() - ladder_dict['bc1']), tdc.coupling_correction_c12],
    [(ladder_dict['b2'].dag() - ladder_dict['b2'])*(ladder_dict['bc2'].dag() - ladder_dict['bc2']), tdc.coupling_correction_2c2],
    [(ladder_dict['bc1'].dag() - ladder_dict['bc1'])*(ladder_dict['bc2'].dag() - ladder_dict['bc2']), tdc.coupling_correction_c1c2]
    ]


opt_params = [['Pulse_c1', 'Amp'], ['Pulse_c2', 'Amp'], ['Pulse_c1', 'slope'], ['Pulse_c2', 'slope'], ['Pulse_c1', 't_gate'], ['Pulse_c2', 't_gate']]          # 3 parameter optimization


################################################################################################################################ 3 parameter optimization ################################################################################################################################

p0 = [-1978261899.97657, -1932390084.17409, 0.00000000450250210723366, 0.00000000387387509766741, 0.0000000334388748965053, 0.000000034634709136852] 


amp_max = -1e9
amp_min = -3e9
t_gate_max = 35e-9
t_gate_min = 10e-9

slope_max = 0.5*t_gate_min
slope_min = 0.1e-9

eta = [1,80,0.1]

upper_bounds = [amp_max, amp_max, slope_max, slope_max, t_gate_max, t_gate_max]
lower_bounds = [amp_min, amp_min, slope_min, slope_min, t_gate_min, t_gate_min]


################################################################################################################################ 3 parameter optimization ################################################################################################################################



assert len(p0) == len(upper_bounds) and len(p0) == len(lower_bounds), 'p0 and bounds have not the same length'



rotating_freq = None

U_perfect = np.matrix(np.diag([1,1,1,-1,1,1,-1,1]))

Parallel_CZ_Gate = {'gate' : '3_Parallel_CZ', 'Hamiltonian' : H, 'Idle_Hamiltonian': H0, 'States' : evec_dict, 'Parameter' : sys_params, 'opt_params' : opt_params, 'starting_values' : p0, 'upper_bounds' : upper_bounds, 'lower_bounds' : lower_bounds, 'rotating_freq' : rotating_freq, 'ladder_dict' : ladder_dict, 'all_vectors' : psi_array, 'comp_eigenstates_lables' : comp_eigenstates_lables, 'U_perfect': U_perfect, 'single_phase_qubits' : single_phase_qubits, 'pauli_op_dict' : pauli_op_dict, 'ft_errors': ft_errors, 'eta' : eta, 'zero_pauli_strings' : zero_pauli_strings } # , 'evec_dict' : evec_dict 'energy_levels' : energy_level, 'num_qubits' : num_qubits }

print(Parallel_CZ_Gate['starting_values'])         # 3,4 parameter Flat-top-Gaussian optimization



args_new, fidelity = hlp.optimize_parameters(Parallel_CZ_Gate)


end_params = {}
for key_a in args_new:
    if key_a not in ['Pulse_c1', 'Pulse_c2']:
        end_params[key_a] = args_new[key_a]
    elif key_a == 'Pulse_c1':
        for key_b in args_new['Pulse_c1']:
            end_params['Pulse_c1' + ': ' + key_b] = args_new['Pulse_c1'][key_b] 
    elif key_a == 'Pulse_c2':
        for key_b in args_new['Pulse_c2']:
            end_params['Pulse_c2' + ': ' + key_b] = args_new['Pulse_c2'][key_b] 

end_params['Fidelity'] = fidelity

end_params['eta'] = eta

if not os.path.exists(path + '/Finished_optimization_results/' + timestr): 
    os.makedirs(path + '/Finished_optimization_results/' + timestr) 

if not os.path.exists(path + '/Finished_optimization_results/' + timestr + '/error_analysis/'): 
    os.makedirs(path + '/Finished_optimization_results/' + timestr + '/error_analysis/') 

if not os.path.exists(path + '/Finished_optimization_results/' + timestr + '/energy_population/'): 
    os.makedirs(path + '/Finished_optimization_results/' + timestr + '/energy_population/') 

if not os.path.exists(path + '/Finished_optimization_results/' + timestr + '/process_unitary/'): 
    os.makedirs(path + '/Finished_optimization_results/' + timestr + '/process_unitary/') 

if not os.path.exists(path + '/Finished_optimization_results/' + timestr + '/flux_pulse/'): 
    os.makedirs(path + '/Finished_optimization_results/' + timestr + '/flux_pulse/') 

hlp.copy_files(path_temp,path + '/Finished_optimization_results/' + timestr)


df_end_params = pd.DataFrame(end_params.items(), columns=['Parameter', 'Value'])  
df_end_params.to_csv(path + '/Finished_optimization_results/' + timestr + '/end_params' + '.csv')


np.save(path + '/Finished_optimization_results/' + timestr + '/end_params', end_params)

print('Script finished')

