import numpy as np
from qutip import *
import itertools

def get_bitstrings(qubits,energy_level_bitstring):


    bitstring_list = list(map(''.join, itertools.product(energy_level_bitstring,repeat = qubits)))

    return bitstring_list

def get_operator_name(qubits_list,cr_an_list):
    operator_name_list = []

    for i in range(len(qubits_list)):
        operator_name_list.append(str(cr_an_list[0]) + str(qubits_list[i]))
    return operator_name_list

def get_vector(qubits,energy_level_bitstring,energy_level,qubits_list,ladder_dict,cr_an_list):
    bitstring = get_bitstrings(qubits,energy_level_bitstring)

    locals().update(ladder_dict)



    bitstring_dict = {}
    bitstring_list = []
    bitstring_name_psi_list = []
    bitstring_name_list = []

    for idx in range(len(bitstring)):
        psi_gnd_temp = basis(energy_level,0)
        for qubit_number in range(qubits-1):
            psi_gnd_temp = tensor(psi_gnd_temp,basis(energy_level,0))

        
        bitstring_name_psi = 'psi_' + bitstring[idx]
        bitstring_name = bitstring[idx]

        for excitation in range(1,energy_level):
            for position in range(0,qubits):
                if bitstring[idx].find(str(excitation),position) == position:
                    for multiply in range(0,excitation): 
                        psi_gnd_temp = eval(cr_an_list[0] + str(qubits_list[position])).dag()*psi_gnd_temp


        bitstring_dict[bitstring_name_psi] = psi_gnd_temp.unit()

        bitstring_list.append(psi_gnd_temp)

        bitstring_name_psi_list.append(bitstring_name_psi)
        bitstring_name_list.append(bitstring_name)
        

    return bitstring_dict, bitstring_list, bitstring_name_psi_list, bitstring_name_list 

