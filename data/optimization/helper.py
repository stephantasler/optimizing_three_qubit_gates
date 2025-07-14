import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import itertools
import copy
import multiprocessing as mp
import functools
import time
from tqdm import tqdm
import scipy as sc
import pandas as pd
import sys 
import shutil
import os


import time_dependence_correction as tdc


path = './Plots/'
path_temp = './Plots/Optimization_temp/'

e = 1.602176634e-19
h = 6.62607015e-34

###################################################################################### Vector Generation ######################################################################################


def get_bitstrings(qubits,energy_level_bitstring):


    bitstring_list = list(map(''.join, itertools.product(energy_level_bitstring,repeat = qubits)))

    return bitstring_list

def get_operator_name(qubits_list,cr_an_list):
    operator_name_list = []

    for i in range(len(qubits_list)):
        operator_name_list.append(str(cr_an_list[0]) + str(qubits_list[i]))
    return operator_name_list

def get_vector(qubits,energy_level_bitstring,energy_level,energy_level_max,qubits_list,ladder_dict,cr_an_list):
    bitstring = get_bitstrings(qubits,energy_level_bitstring)
    locals().update(ladder_dict)



    bitstring_dict = {}
    bitstring_list = []
    bitstring_name_psi_list = []
    bitstring_name_list = []

    for idx in range(len(bitstring)):
        psi_gnd_temp = basis(energy_level[0],0)
        for qubit_number in range(qubits-1):
            psi_gnd_temp = tensor(psi_gnd_temp,basis(energy_level[qubit_number+1],0))

        
        bitstring_name_psi = 'psi_' + bitstring[idx]
        bitstring_name = bitstring[idx]

        for excitation in range(1,energy_level_max):
            for position in range(0,qubits):
                if bitstring[idx].find(str(excitation),position) == position:
                    for multiply in range(0,excitation): 
                        psi_gnd_temp = eval(cr_an_list[0] + str(qubits_list[position])).dag()*psi_gnd_temp


        bitstring_dict[bitstring_name_psi] = psi_gnd_temp.unit()

        bitstring_list.append(psi_gnd_temp)

        bitstring_name_psi_list.append(bitstring_name_psi)
        bitstring_name_list.append(bitstring_name)
        

    return bitstring_dict, bitstring_list, bitstring_name_psi_list, bitstring_name_list 


##################################################################################### Ladder Generation #####################################################################################

def get_creation_annihilation(qubits_list, cr_an_list, energy_level):
    dict={}
    qeye_list=[]
    for idx in range(len(qubits_list)):
        qeye_list.append(qeye(energy_level[idx]))
    i=0       
    for entry in qubits_list:
        dummy_list = copy.copy(qeye_list)
        key = cr_an_list[0] + entry
        dummy_list[i] = destroy(energy_level[i])
        dict[key] = tensor(dummy_list)
        i+=1
    return dict

##################################################################################### Pauli String Generation #####################################################################################

def get_pauli_string_operator(pauli_label_list):                                                                # pauli string operator creation function
    pauli_string_list = []
    for label in pauli_label_list:
        pauli_dummy_list = []
        for i in range(len(label)):
            if label[i] == 'i':
                pauli_dummy_list.append(qeye(2))
            if label[i] == 'x':
                pauli_dummy_list.append(sigmax())
            if label[i] == 'y':
                pauli_dummy_list.append(sigmay())
            if label[i] == 'z':
                pauli_dummy_list.append(sigmaz())
        pauli_string_list.append(tensor(pauli_dummy_list))
    return(pauli_string_list)


def get_pauli_op_dict(num_qubits):
    #op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()]] * num_qubits
    op_label = [["i", "x", "y", "z"]] * num_qubits

    op_label_all = []

    for i in range(len(op_label[0])):                                                                       # creation of all string arrays
        for j in range(len(op_label[1])):
            for k in range(len(op_label[2])):
                op_label_all.append(op_label[0][i] + op_label[1][j] + op_label[2][k])

    pauli_string_operator = get_pauli_string_operator(op_label_all)                                                  # pauli string operator creation

    pauli_op_dict = {}

    for idx in range(len(op_label_all)):                                                                     # pauli string operator dictionary creation (all pauli operators)
        pauli_op_dict[str(op_label_all[idx])] = pauli_string_operator[idx] 
    
    return(pauli_op_dict)


############################################################################### Coupling Parameter Calculation ###############################################################################
def get_coupling(frequency,C_dict,w_dict,C_mat,nodes):

    node_mapping = {}
    for i in range(len(nodes)):
        node_mapping[nodes[i]] = i

    hbar = h/(2*np.pi)

    C_inv = np.linalg.inv(C_mat)
    
    EC = ((e)**2/(2*h))*C_inv

    params = {}

    params['EC'] = EC
    
    if frequency == 'EJ':
        idx = 0
        for key in w_dict:
            params['EJ' + nodes[idx]] = w_dict[key]
            params['w' + nodes[idx]] = np.sqrt(8 * EC[idx,idx] * w_dict[key]) - EC[idx,idx]
            idx += 1

    
    elif frequency == 'omega':
        idx = 0
        for key in w_dict:
            params['w' + nodes[idx]] = w_dict[key]
            params['EJ' + nodes[idx]] = ((1/8)*(w_dict[key]+EC[idx,idx])**2)/EC[idx,idx]
            idx += 1

    idx = 0
    for node in nodes:
        params['a' + node] = -EC[idx,idx]
        params['zeta' + node] = np.sqrt(8*(EC[idx,idx]/params['EJ'+ node]))
        idx += 1

    coupling = {}
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            params['g'+nodes[i]+nodes[j]] = -2*(EC[i,j]+EC[j,i])*(1/(np.sqrt(params['zeta'+ nodes[i]]*params['zeta' + nodes[j]])))

    return params

#################################################################################### Eigenstate Calculation ####################################################################################

def get_eigenstate(H, psi_array_temp, psi_array_label, energy_level, Multi):
    if Multi == True:
        
        val, vec = np.linalg.eigh(H.full())
        ev = [None]*len(psi_array_temp)
        eval = [None]*len(psi_array_temp)
        overlap = [None]*len(psi_array_temp)
        label = [None]*len(psi_array_temp)

        ev_dict = {}
        eval_dict = {}

        for element in range(len(psi_array_temp)):
            state_mat = psi_array_temp[element].unit().full()
            qualityfactor = 0
            pos_eig = 0
            for i in range(np.prod(energy_level)):
                q = np.abs(np.dot(state_mat.conj().T, vec[:, i]))**2
                if q >= qualityfactor:
                    qualityfactor = q
                    pos_eig = i
            ev_dict[psi_array_label[element]] = Qobj(vec[:, pos_eig], dims = psi_array_temp[0].dims)        
            eval_dict[psi_array_label[element]] = val[pos_eig]                                               
            label[element] = psi_array_label[element]                                               

            ev[element] = Qobj(vec[:, pos_eig], dims = psi_array_temp[0].dims)
            eval[element] = val[pos_eig]
            overlap[element] = float(np.abs(np.dot(state_mat.T,Qobj(ev[element]).full()))**2)
        return(ev, eval, overlap, ev_dict, eval_dict, label)

    elif Multi == False:
        state_mat = np.matrix(psi_array_temp.unit())
        val, vec = np.linalg.eigh(H)
        qualityfactor = 0
        pos_eig = 0
        for i in range(np.prod(energy_level)):
            q = np.abs(np.dot(state_mat.H, vec[:, i]))**2
            if q >= qualityfactor:
                qualityfactor = q
                pos_eig = i
        ev = Qobj(vec[:, pos_eig], dims = dims(psi_array_temp))
        eval = val[pos_eig]
        
        overlap = float(np.abs(np.dot(state_mat.T,np.matrix(Qobj(ev))))**2)
        return(ev,eval, overlap, None, None, None)
#################################################################################### Hamiltonian Calculation ####################################################################################
    
def get_H(params, ladder_dict):

    H = params['w1'] * ladder_dict['b1'].dag() * ladder_dict['b1'] + params['w2'] * ladder_dict['b2'].dag() * ladder_dict['b2'] + params['w3'] * ladder_dict['b3'].dag() * ladder_dict['b3'] +  params['wc1'] * ladder_dict['bc1'].dag() * ladder_dict['bc1'] + params['wc2'] * ladder_dict['bc2'].dag() * ladder_dict['bc2']\
    + params['a1']/2 * ladder_dict['b1'].dag() * ladder_dict['b1'].dag() * ladder_dict['b1'] * ladder_dict['b1'] + params['a2']/2 * ladder_dict['b2'].dag() * ladder_dict['b2'].dag() * ladder_dict['b2'] * ladder_dict['b2']\
    + params['ac1']/2 * ladder_dict['bc1'].dag() * ladder_dict['bc1'].dag() * ladder_dict['bc1'] * ladder_dict['bc1'] + params['ac2']/2 * ladder_dict['bc2'].dag() * ladder_dict['bc2'].dag() * ladder_dict['bc2'] * ladder_dict['bc2']\
    + params['a3']/2 * ladder_dict['b3'].dag() * ladder_dict['b3'].dag() * ladder_dict['b3'] * ladder_dict['b3']\
    + params['g13'] * (ladder_dict['b1'] * ladder_dict['b3'] - ladder_dict['b1'] * ladder_dict['b3'].dag() - ladder_dict['b1'].dag() * ladder_dict['b3'] + ladder_dict['b1'].dag() * ladder_dict['b3'].dag())\
    + params['g1c2'] * (ladder_dict['b1'] * ladder_dict['bc2'] - ladder_dict['b1'] * ladder_dict['bc2'].dag() - ladder_dict['b1'].dag() * ladder_dict['bc2'] + ladder_dict['b1'].dag() * ladder_dict['bc2'].dag())\
    + params['gc13'] * (ladder_dict['b3'] * ladder_dict['bc1'] - ladder_dict['b3'] * ladder_dict['bc1'].dag() - ladder_dict['b3'].dag() * ladder_dict['bc1'] + ladder_dict['b3'].dag() * ladder_dict['bc1'].dag())\
    + params['g1c1'] * (ladder_dict['b1'] * ladder_dict['bc1'] - ladder_dict['b1'] * ladder_dict['bc1'].dag() - ladder_dict['b1'].dag() * ladder_dict['bc1'] + ladder_dict['b1'].dag() * ladder_dict['bc1'].dag())\
    + params['gc23'] * (ladder_dict['b3'] * ladder_dict['bc2'] - ladder_dict['b3'] * ladder_dict['bc2'].dag() - ladder_dict['b3'].dag() * ladder_dict['bc2'] + ladder_dict['b3'].dag() * ladder_dict['bc2'].dag())\
    + params['gc1c2'] * (ladder_dict['bc1'] * ladder_dict['bc2'] - ladder_dict['bc1'] * ladder_dict['bc2'].dag() - ladder_dict['bc1'].dag() * ladder_dict['bc2'] + ladder_dict['bc1'].dag() * ladder_dict['bc2'].dag())\
    + params['gc12'] * (ladder_dict['b2'] * ladder_dict['bc1'] - ladder_dict['b2'] * ladder_dict['bc1'].dag() - ladder_dict['b2'].dag() * ladder_dict['bc1'] + ladder_dict['b2'].dag() * ladder_dict['bc1'].dag())\
    + params['g2c2'] * (ladder_dict['b2'] * ladder_dict['bc2'] - ladder_dict['b2'] * ladder_dict['bc2'].dag() - ladder_dict['b2'].dag() * ladder_dict['bc2'] + ladder_dict['b2'].dag() * ladder_dict['bc2'].dag())\
    + params['g12'] * (ladder_dict['b1'] * ladder_dict['b2'] - ladder_dict['b1'] * ladder_dict['b2'].dag() - ladder_dict['b1'].dag() * ladder_dict['b2'] + ladder_dict['b1'].dag() * ladder_dict['b2'].dag())\
    + params['g23'] * (ladder_dict['b3'] * ladder_dict['b2'] - ladder_dict['b3'] * ladder_dict['b2'].dag() - ladder_dict['b3'].dag() * ladder_dict['b2'] + ladder_dict['b3'].dag() * ladder_dict['b2'].dag())

    return(2*np.pi*H)


######################################################################### Optimization #########################################################################


def Mesolve_func(idx, args_me):
    output = mesolve(H = args_me[idx][0], rho0 = args_me[idx][1], tlist = args_me[idx][2], args = args_me[idx][3], options = Options(nsteps = 1e8, atol=1e-16, rtol=1e-16), progress_bar = False)
    return(output)

#################################################################################### Unitary Calculation ####################################################################################


def get_U(H, H0, input_states, params, comp_eigenstates_lables, ladder_dict, all_vectors_array, rotating_freq = None, timesteps = 1000, plot = False):
    if rotating_freq == None:
        t_gate = np.max([params['Pulse_c1']['t_gate'],params['Pulse_c2']['t_gate']])                                            # 3,4 parameter Flat-top-Gaussian optimization

        tau = np.linspace(0, t_gate, timesteps)


        args_me=[]
        for i in range(len(comp_eigenstates_lables)):
            args_me.append([H,input_states[comp_eigenstates_lables[i]], tau, params])

        pool = mp.Pool(mp.cpu_count()-1)
        mesolve_result = pool.map(functools.partial(Mesolve_func, args_me = args_me), range(0,len(input_states)))
        pool.close()



        if plot == True:

            Pulse1_temp = []
            Pulse2_temp = []

            for t in tau:
                Pulse1_temp.append(tdc.Flux_pulse_c1(t, params))
                Pulse2_temp.append(tdc.Flux_pulse_c2(t, params))
            
            fig = plt.figure(figsize = (40, 20))
            plt.plot(tau*1e9, Pulse1_temp, label = 'Pulse_1')
            plt.plot(tau*1e9, Pulse2_temp, label = 'Pulse_2')
            plt.xlabel('Time in ns')
            plt.legend()
            plt.ylabel('Pulses')
            plt.savefig(path_temp + 'flux_pulses.pdf')


            populations = np.zeros((len(input_states),timesteps))

            for states in range(len(input_states)):
                for time in range(timesteps):
                    populations[states, time] = expect(input_states[comp_eigenstates_lables[states]]*input_states[comp_eigenstates_lables[states]].dag(),mesolve_result[states].states[time])


            fig = plt.figure(figsize = (40, 20))
            for i in range(0,np.shape(populations)[0]):
                plt.plot(tau*1e9, populations[i, :], label = comp_eigenstates_lables[i])
            plt.xlabel('Time in ns')
            plt.ylim(-0.1, 1.1)
            plt.legend()
            plt.ylabel('Overlap with initial state')
            plt.savefig(path_temp + 'populations.pdf')



        states_t = [None]*len(input_states)


        U = np.zeros((len(input_states), len(input_states)), dtype = 'complex')
        U_abs = np.zeros((len(input_states), len(input_states)), dtype = 'complex')


        for i in range(len(input_states)):
            for j in range(len(input_states)):
                U[i, j] = complex((input_states[comp_eigenstates_lables[i]].dag()*mesolve_result[j].states[-1]))


        for i in range(len(input_states)):
            for j in range(len(input_states)):
                U_abs[i, j] = np.abs(np.sqrt(complex((input_states[comp_eigenstates_lables[i]].dag()*mesolve_result[j].states[-1]))**2))

        return np.matrix(U), np.matrix(U_abs)


def get_phase_gate(phase_params):
    phi1 = phase_params[0]
    phi2 = phase_params[1]
    phi3 = phase_params[2]

    return(np.diag([np.exp(-1j*(phi1+phi2+phi3)/2),
                    np.exp(-1j*(phi1+phi2-phi3)/2),
                    np.exp(-1j*(phi1-phi2+phi3)/2),
                    np.exp(-1j*(phi1-phi2-phi3)/2),
                    np.exp(1j*(phi1-phi2-phi3)/2),
                    np.exp(1j*(phi1-phi2+phi3)/2),
                    np.exp(1j*(phi1+phi2-phi3)/2),
                    np.exp(1j*(phi1+phi2+phi3)/2)]))

def calc_fid(phase_params,args):
    U_corrected = np.matmul(get_phase_gate(phase_params), args['U'])
    fid = np.abs(np.trace(np.matmul(np.conj(U_corrected.T), args['U_perfect'])))/len(args['U'])

    np.save(path + 'U_corr.npy', U_corrected)

    return(fid)

def calc_leak(phase_params,args):
    U_corrected = np.matmul(get_phase_gate(phase_params), args['U'])
    leak = np.abs(np.trace(np.matmul(np.conj(U_corrected.T), U_corrected)))/len(args['U'])

    return(leak)




def get_fidelity(U, U_perfect, single_phase_qubits):
    initial_params = [1 for i in range(single_phase_qubits)]
    bounds = [(0,2*np.pi) for i in range(single_phase_qubits)]

    args = {}
    args['U'] = U
    args['U_perfect'] = U_perfect

    def phase_cost_function(phase_params,args):
    
        U_corrected = np.matmul(get_phase_gate(phase_params), args['U'])
        fid_gate = np.abs(np.trace(np.matmul(np.conj(U_corrected.T), args['U_perfect'])))/len(args['U'])
        fid_leakage = np.abs(np.trace(np.matmul(np.conj(U_corrected.T), U_corrected)))/len(args['U'])

        fid = (fid_gate + fid_leakage)/2
        return (1-fid)
    
    result = sc.optimize.minimize(fun = phase_cost_function, x0 = initial_params, args = args, bounds = bounds, method = 'Powell', options={'fatol': 1e-15, 'maxfev': 40000, 'disp': True, 'adaptive': True})
    optimized_params = result.x

    print(optimized_params)
    fidelity = calc_fid(optimized_params,args)
    leakage = calc_leak(optimized_params,args)

    U_corrected = np.matmul(get_phase_gate(optimized_params), args['U'])


    return fidelity, optimized_params, U_corrected, leakage


def get_error_model(U_sim, U_ideal, pauli_op_dict, zero_pauli_strings):


    global_phase_corr = np.angle(np.diagonal(U_sim))[0]
    print(global_phase_corr)
    rotation_matrix = np.diag([np.exp(-1j*global_phase_corr),np.exp(-1j*global_phase_corr),np.exp(-1j*global_phase_corr),np.exp(-1j*global_phase_corr),np.exp(-1j*global_phase_corr),np.exp(-1j*global_phase_corr),np.exp(-1j*global_phase_corr),np.exp(-1j*global_phase_corr)])
    U_sim_rot = np.matmul(rotation_matrix,U_sim)


    U_err = np.matmul(U_sim_rot, np.linalg.inv(U_ideal))                                                        # calculation U_err 

    U_pa_array = {}
    U_all_array = {}
    U_pa_array_all_err = {}
    U_pa_zero = {}

    for key in pauli_op_dict:
        U_all_array[str(key)] = np.abs(np.trace(np.matmul(np.conj(np.array(pauli_op_dict[key].full()).T), U_err))/len(U_err))**2
        if str(key)[0] != 'i' and str(key)[2] != 'i' and str(key) != 'ziz' and str(key) != 'zzz':                                           #only focuses on noise on data qubits and ignors ziz and zzz, since less significant for QEC performance
            U_pa_array[str(key)] = np.abs(np.trace(np.matmul(np.conj(np.array(pauli_op_dict[key].full()).T), U_err))/len(U_err))**2
        if str(key) != 'iii':                                           #only focuses on noise on data qubits and ignors ziz and zzz, since less significant for QEC performance
            U_pa_array_all_err[str(key)] = np.abs(np.trace(np.matmul(np.conj(np.array(pauli_op_dict[key].full()).T), U_err))/len(U_err))**2
        if str(key) in zero_pauli_strings:
            U_pa_zero[str(key)] = np.abs(np.trace(np.matmul(np.conj(np.array(pauli_op_dict[key].full()).T), U_err))/len(U_err))**2

    return(U_pa_array, U_all_array, U_pa_array_all_err, U_pa_zero)



#Optimizer
def cost_function(opt_params, Gate):
    starttime = time.time()
    for i in range(len(Gate['opt_params'])):
        Gate['Parameter'][Gate['opt_params'][i][0]][Gate['opt_params'][i][1]] = opt_params[i]

    print('')
    print('------------------------------------------------------------------------------------')
    print('')
    
    print('Optimierte Parameter Start')
    mid_params = {}
    for i in range(len(Gate['opt_params'])):
        print(Gate['opt_params'][i][0],Gate['opt_params'][i][1])              # 3,4 parameter Flat-top-Gaussian optimization
        print(opt_params[i])
        mid_params[Gate['opt_params'][i][0] + ': ' + Gate['opt_params'][i][1]] = opt_params[i]


    
    print('Optimierte Parameter Ende')

    U, U_abs = get_U(H = Gate['Hamiltonian'], H0 = Gate['Idle_Hamiltonian'], input_states = Gate['States'], params = Gate['Parameter'], comp_eigenstates_lables = Gate['comp_eigenstates_lables'], rotating_freq = Gate['rotating_freq'], ladder_dict = Gate['ladder_dict'],  all_vectors_array = Gate['all_vectors'], plot = True) #  , Gate['energy_levels'], Gate['num_qubits'])
    
    
    fidelity = get_fidelity(U, Gate['U_perfect'], Gate['single_phase_qubits'])[0]
    U_corrected = get_fidelity(U, Gate['U_perfect'], Gate['single_phase_qubits'])[2]
    leakage = get_fidelity(U, Gate['U_perfect'], Gate['single_phase_qubits'])[3]

    #print(Gate['ft_errors'])

    old_ft_breaking_error_model, full_pauli_model, even_error_model, zero_error_model  = get_error_model(U_corrected, Gate['U_perfect'], Gate['pauli_op_dict'], Gate['zero_pauli_strings'])

    full_error_model_ucl = {k.upper() : v for k,v in full_pauli_model.items()}                      #Upper case letter

    ft_errors = sum(full_error_model_ucl[key]*Gate['ft_errors'][key] for key in Gate['ft_errors'])

    print('ft_errors')
    print(ft_errors)



    old_labels_ft_breaking, old_values_ft_breaking = old_ft_breaking_error_model.keys(), old_ft_breaking_error_model.values()
    old_data_qubit_ft_braking_error_model_sum = sum(old_ft_breaking_error_model.values())

    labels_even, values_even = even_error_model.keys(), even_error_model.values()
    data_qubit_even_error_model_sum = sum(even_error_model.values())

    labels_zero, values_zero = zero_error_model.keys(), zero_error_model.values()
    data_qubit_zero_error_model_sum = sum(zero_error_model.values())




    print('Error model sum:')
    print(old_data_qubit_ft_braking_error_model_sum)

    print('Full error model sum:')
    print(data_qubit_even_error_model_sum)

    spacing_factor = 1

    x_ft_breaking = np.arange(len(old_labels_ft_breaking)) * spacing_factor
    x_even = np.arange(len(labels_even)) * spacing_factor
    x_zero = np.arange(len(labels_zero)) * spacing_factor

###################################################################################################################################################################################################################################################################################

    plt.figure( figsize = (30,10))                                                                                                                              # plotting of ft breaking pauli weights
    plt.hlines(y=1e-4, xmin = -100, xmax = np.max(x_ft_breaking)+100, colors= 'black', linestyles= 'dotted', linewidth = 0.5 )
    plt.hlines(y=1e-5, xmin = -100, xmax = np.max(x_ft_breaking)+100, colors= 'black', linestyles= 'dotted', linewidth = 0.5 )
    plt.hlines(y=1e-6, xmin = -100, xmax = np.max(x_ft_breaking)+100, colors= 'black', linestyles= 'dotted', linewidth = 0.5 )
    plt.xlim(left=min(x_ft_breaking) - 0.5, right=max(x_ft_breaking) + 0.5)
    plt.bar(x_ft_breaking, old_values_ft_breaking)
    plt.xticks(x_ft_breaking,old_labels_ft_breaking)
    plt.title('FT-braking Pauli Error Channel')
    plt.ylabel('amplitude')
    plt.tick_params(axis='x', labelrotation=90)
    plt.yscale('log')
    plt.ylim(bottom=1e-10)
    plt.grid(color = 'tab:grey', ls = '--',)

    plt.savefig(path_temp + 'old_ft_breaking_error_channels.pdf')

    df_error_values = pd.DataFrame(old_ft_breaking_error_model.items(), columns=['Key', 'Value'])  
    df_error_values.to_csv(path_temp + '/old_ft_breaking_error_vals_temp.csv')

###################################################################################################################################################################################################################################################################################
    
    plt.figure( figsize = (30,10))                                                                                                                              # plotting of zero pauli weights
    plt.hlines(y=1e-4, xmin = -100, xmax = np.max(x_zero)+100, colors= 'black', linestyles= 'dotted', linewidth = 0.5 )
    plt.hlines(y=1e-5, xmin = -100, xmax = np.max(x_zero)+100, colors= 'black', linestyles= 'dotted', linewidth = 0.5 )
    plt.hlines(y=1e-6, xmin = -100, xmax = np.max(x_zero)+100, colors= 'black', linestyles= 'dotted', linewidth = 0.5 )
    plt.xlim(left=min(x_zero) - 0.5, right=max(x_zero) + 0.5)
    plt.bar(x_zero, values_zero)
    plt.xticks(x_zero,labels_zero)
    plt.title('FT-braking Pauli Error Channel')
    plt.ylabel('amplitude')
    plt.tick_params(axis='x', labelrotation=90)
    plt.yscale('log')
    plt.ylim(bottom=1e-10)
    plt.grid(color = 'tab:grey', ls = '--',)

    plt.savefig(path_temp + 'zero_error_channels.pdf')

    df_error_values = pd.DataFrame(zero_error_model.items(), columns=['Key', 'Value'])  
    df_error_values.to_csv(path_temp + '/zero_error_vals_temp.csv')

###################################################################################################################################################################################################################################################################################

    plt.figure( figsize = (30,10))                                                                                                                              # plotting of even pauli weights
    plt.hlines(y=1e-4, xmin = -100, xmax = np.max(x_even)+100, colors= 'black', linestyles= 'dotted', linewidth = 0.5 )
    plt.hlines(y=1e-5, xmin = -100, xmax = np.max(x_even)+100, colors= 'black', linestyles= 'dotted', linewidth = 0.5 )
    plt.hlines(y=1e-6, xmin = -100, xmax = np.max(x_even)+100, colors= 'black', linestyles= 'dotted', linewidth = 0.5 )
    plt.xlim(left=min(x_even) - 0.5, right=max(x_even) + 0.5)
    plt.bar(x_even, values_even)
    plt.xticks(x_even,labels_even)
    plt.title('Even Pauli Error Channel')
    plt.ylabel('amplitude')
    plt.tick_params(axis='x', labelrotation=90)
    plt.yscale('log')
    plt.ylim(bottom=1e-10)
    plt.grid(color = 'tab:grey', ls = '--',)

    plt.savefig(path_temp + 'even_error_channels.pdf')


    df_error_values_all = pd.DataFrame(even_error_model.items(), columns=['Key', 'Value'])  
    df_error_values_all.to_csv(path_temp + '/even_error_vals_temp.csv')


    endtime = time.time()

    mid_params['Fidelity'] = fidelity
    mid_params['Error'] = 1-fidelity
    mid_params['Leakage'] = leakage

    print('Fidelity')
    print(fidelity)

    print('Error')
    print((1-fidelity))

    print('Leakage')
    print(leakage)

    Gate['opt_error'].append(1-fidelity)
    Gate['opt_error_leakage'].append(1-leakage)


    step_num = len(Gate['opt_error'])

    steps = np.linspace(1,step_num,step_num)

    fig = plt.figure(figsize = (40, 20))
    plt.plot(steps, Gate['opt_error'], label = '1 - Fidelity')
    plt.plot(steps, Gate['opt_error_leakage'], label = '1 - Leakage')
    plt.xlabel('steps')
    plt.legend()
    plt.ylabel('Error')
    plt.yscale('log')
    plt.savefig(path_temp + 'optimization_cost.pdf')



    print('cost_function')
    print(f"Time taken {endtime-starttime} seconds")
    print(f"Time taken {((endtime-starttime)-((endtime-starttime)%60))/60} minutes and {(endtime-starttime)%60} seconds")


    


    eta = Gate['eta']

    mid_params['eta'] = eta

    df_mid_params = pd.DataFrame(mid_params.items(), columns=['Parameter', 'Value'])  
    df_mid_params.to_csv(path_temp + '/opt_params_temp.csv')
    return (eta[0]*(1-leakage) + eta[1] * ft_errors + eta[2]* data_qubit_even_error_model_sum)

def optimize_parameters(Gate):

    Gate['opt_error'] = []
    Gate['opt_error_leakage'] = []

    initial_params = Gate['starting_values']
    lower_bounds = Gate['lower_bounds']
    upper_bounds = Gate['upper_bounds']
    
    bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(Gate['opt_params']))]

    
    result = sc.optimize.minimize(fun = cost_function, x0 = initial_params, args = (Gate), bounds = bounds, method = 'Nelder-Mead', options={'fatol': 1e-6, 'maxfev': 800, 'disp': True, 'adaptive' : True})            # Gaussian-Flat-Top optimization 
    
    
    optimized_params = result.x
    
    for i in range(len(optimized_params)):
        Gate['Parameter'][Gate['opt_params'][i][0]][Gate['opt_params'][i][1]] = optimized_params[i]

    U, U_abs = get_U(H = Gate['Hamiltonian'], H0 = Gate['Idle_Hamiltonian'], input_states = Gate['States'], params = Gate['Parameter'], comp_eigenstates_lables = Gate['comp_eigenstates_lables'], rotating_freq = Gate['rotating_freq'], ladder_dict = Gate['ladder_dict'],  all_vectors_array = Gate['all_vectors'], plot = True) #  , Gate['energy_levels'], Gate['num_qubits'])
    fidelity = get_fidelity(U, Gate['U_perfect'], Gate['single_phase_qubits'])[0]


    
    return Gate['Parameter'], fidelity




def copy_files(source_dir, destination_dir):
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    # Ensure the destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Iterate over files in the source directory
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        
        # Copy only files (not directories)
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
            print(f"Copied: {source_path} -> {destination_path}")



