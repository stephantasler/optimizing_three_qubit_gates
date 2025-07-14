import numpy as np
import math 




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gaussian(x,a,b,c):
    return a*np.exp(-((x-b)**2)/(2*c**2))


########################################################################################## Puls functions ##########################################################################################
def activation_pulse(taulist,args):
    return 1 - sigmoid(args['r']*((taulist)-args['activation'])/args['activation']) + sigmoid(args['r']*((taulist)-(args['activation'] + args['delta']))/args['activation'])

def activation_pulse_int(taulist,args):
    return sigmoid(args['r']*((taulist)-args['activation'])/args['activation']) - sigmoid(args['r']*((taulist)-(args['activation'] + args['delta']))/args['activation'])

############################################### Sigmoid ###############################################

def activation_sigmoid(taulist,args):
    return 1 - sigmoid(args['r']*((taulist)-args['activation'])/args['activation']) 

def activation_sigmoid_int(taulist,args):
    return sigmoid(args['r']*((taulist)-args['activation'])/args['activation']) 


############################################### Recktangular ###############################################

def activation_recktangular(taulist,activation):
    return 1-np.heaviside((taulist-activation),0.5)


def activation_recktangular_int(taulist,activation):
    return np.heaviside((taulist-activation),0.5)



def activation_puls_recktangular(taulist,activation,delta):
    return  1 - np.heaviside((taulist-activation),0.5) + np.heaviside((taulist- delta -activation),0.5)

def activation_puls_recktangular_int(taulist,activation,delta):
    return np.heaviside((taulist - activation),0.5) - np.heaviside((taulist- delta - activation),0.5) 

############################################### PWC-Function ###############################################

def piecewise_constant(t, intervals, values):
    for i, (start, end) in enumerate(zip(intervals[:-1], intervals[1:])):
        if start <= t < end:
            return values[i]
    return values[-1]  



############################################### Flat-Top-Gaussian ###############################################

def Pulse_1(t, args):
    t_diff = (np.max([args['Pulse_c1']['t_gate'],args['Pulse_c2']['t_gate']]) - args['Pulse_c1']['t_gate'])/2
    return(1/4*(1 + math.erf((t-t_diff-args['Pulse_c1']['slope'])/args['Pulse_c1']['slope']))*(1 + math.erf((args['Pulse_c1']['t_gate']-t+t_diff-args['Pulse_c1']['slope'])/args['Pulse_c1']['slope'])))

def Pulse_2(t, args):
    t_diff = (np.max([args['Pulse_c1']['t_gate'],args['Pulse_c2']['t_gate']]) - args['Pulse_c2']['t_gate'])/2
    return(1/4*(1 + math.erf((t-t_diff-args['Pulse_c1']['slope'])/args['Pulse_c2']['slope']))*(1 + math.erf((args['Pulse_c2']['t_gate']-t+t_diff-args['Pulse_c1']['slope'])/args['Pulse_c2']['slope'])))


############################################################################### Flux Pulses Setup ###############################################################################

#Flux pulse on Coupler 1
def Flux_pulse_c1(t, args):
    if 'Sigmoid' in args['Pulse_c1']:
        return(2*np.pi*args['Pulse_c1']['Amp'] * activation_pulse_int(t, args))
    elif 'Erfc' in args['Pulse_c1']:
        return(2*np.pi*args['Pulse_c1']['Amp'] * Pulse_1(t, args))

#Flux pulse on Coupler 2
def Flux_pulse_c2(t, args):
    if 'Sigmoid' in args['Pulse_c2']:
        return(2*np.pi*args['Pulse_c2']['Amp'] * activation_pulse_int(t, args))
    elif 'Erfc' in args['Pulse_c2']:
        return(2*np.pi*args['Pulse_c2']['Amp'] * Pulse_2(t, args))

#Flux pulse on Coupler 3
def Flux_pulse_c3(t, args):
    if 'Sigmoid' in args['Pulse_c3']:
        return(2*np.pi*args['Pulse_c3']['Amp'] * activation_pulse_int(t, args))
    elif 'Erfc' in args['Pulse_c3']:
        return(2*np.pi*args['Pulse_c3']['Amp'] * Pulse_3(t, args))


############################################################################### Timedependent Couplings Setup ###############################################################################


def coupling_correction_1c1(t, args):
    Ampc1 = 0

    if 'Sigmoid' in args['Pulse_c1']:
        Ampc1 = args['Pulse_c1']['Amp'] * activation_pulse_int(t, args)
    elif 'Erfc' in args['Pulse_c1']:
        Ampc1 = args['Pulse_c1']['Amp'] * Pulse_1(t, args)


    g_idle = -0.5*args['EC'][0,1]*((args['EC'][0,0]/(args['w1']+args['EC'][0,0]))*(args['EC'][1,1]/(args['wc1']+args['EC'][1,1])))**(-1/2)
    g_pulse = -0.5*args['EC'][0,1]*((args['EC'][0,0]/(args['w1']+args['EC'][0,0]))*(args['EC'][1,1]/(args['wc1']+Ampc1+args['EC'][1,1])))**(-1/2)


    return(2*np.pi*(g_pulse-g_idle))



def coupling_correction_c12(t,args):
    Ampc1 = 0

    if 'Sigmoid' in args['Pulse_c1']:
        Ampc1 = args['Pulse_c1']['Amp'] * activation_pulse_int(t, args)
    elif 'Erfc' in args['Pulse_c1']:
        Ampc1 = args['Pulse_c1']['Amp'] * Pulse_1(t, args)

    g_idle = -0.5*args['EC'][2,1]*((args['EC'][2,2]/(args['w2']+args['EC'][2,2]))*(args['EC'][1,1]/(args['wc1']+args['EC'][1,1])))**(-1/2)
    g_pulse = -0.5*args['EC'][2,1]*((args['EC'][2,2]/(args['w2']+args['EC'][2,2]))*(args['EC'][1,1]/(args['wc1']+Ampc1+args['EC'][1,1])))**(-1/2)
    

    return(2*np.pi*(g_pulse-g_idle))



def coupling_correction_c1c2(t,args):
    Ampc1 = 0
    Ampc2 = 0

    if 'Sigmoid' in args['Pulse_c1']:
        Ampc1 = args['Pulse_c1']['Amp'] * activation_pulse_int(t, args)
    elif 'Erfc' in args['Pulse_c1']:
        Ampc1 = args['Pulse_c1']['Amp'] * Pulse_1(t, args)

    if 'Sigmoid' in args['Pulse_c2']:
        Ampc2 = args['Pulse_c2']['Amp'] * activation_pulse_int(t, args)
    elif 'Erfc' in args['Pulse_c2']:
        Ampc2 = args['Pulse_c2']['Amp'] * Pulse_2(t, args)

    g_idle = -0.5*args['EC'][1,3]*((args['EC'][1,1]/(args['wc1']+args['EC'][1,1]))*(args['EC'][3,3]/(args['wc2']+args['EC'][3,3])))**(-1/2)
    g_pulse = -0.5*args['EC'][1,3]*((args['EC'][1,1]/(args['wc1']+Ampc1+args['EC'][1,1]))*(args['EC'][3,3]/(args['wc2']+Ampc2+args['EC'][3,3])))**(-1/2)

    return(2*np.pi*(g_pulse-g_idle))



def coupling_correction_c13(t,args):
    Ampc1 = 0

    if 'Sigmoid' in args['Pulse_c1']:
        Ampc1 = args['Pulse_c1']['Amp'] * activation_pulse_int(t, args)
    elif 'Erfc' in args['Pulse_c1']:
        Ampc1 = args['Pulse_c1']['Amp'] * Pulse_1(t, args)

    g_idle = -0.5*args['EC'][4,1]*((args['EC'][4,4]/(args['w3']+args['EC'][4,4]))*(args['EC'][1,1]/(args['wc1']+args['EC'][1,1])))**(-1/2)
    g_pulse = -0.5*args['EC'][4,1]*((args['EC'][4,4]/(args['w3']+args['EC'][4,4]))*(args['EC'][1,1]/(args['wc1']+Ampc1+args['EC'][1,1])))**(-1/2)
    

    return(2*np.pi*(g_pulse-g_idle))


def coupling_correction_1c2(t, args):
    Ampc2 = 0

    if 'Sigmoid' in args['Pulse_c2']:
        Ampc2 = args['Pulse_c2']['Amp'] * activation_pulse_int(t, args)
    elif 'Erfc' in args['Pulse_c2']:
        Ampc2 = args['Pulse_c2']['Amp'] * Pulse_2(t, args)


    g_idle = -0.5*args['EC'][0,3]*((args['EC'][0,0]/(args['w1']+args['EC'][0,0]))*(args['EC'][3,3]/(args['wc2']+args['EC'][3,3])))**(-1/2)
    g_pulse = -0.5*args['EC'][0,3]*((args['EC'][0,0]/(args['w1']+args['EC'][0,0]))*(args['EC'][3,3]/(args['wc2']+Ampc2+args['EC'][3,3])))**(-1/2)

    return(2*np.pi*(g_pulse-g_idle))


def coupling_correction_2c2(t, args):
    Ampc2 = 0

    if 'Sigmoid' in args['Pulse_c2']:
        Ampc2 = args['Pulse_c2']['Amp'] * activation_pulse_int(t, args)
    elif 'Erfc' in args['Pulse_c2']:
        Ampc2 = args['Pulse_c2']['Amp'] * Pulse_2(t, args)


    g_idle = -0.5*args['EC'][2,3]*((args['EC'][2,2]/(args['w2']+args['EC'][2,2]))*(args['EC'][3,3]/(args['wc2']+args['EC'][3,3])))**(-1/2)
    g_pulse = -0.5*args['EC'][2,3]*((args['EC'][2,2]/(args['w2']+args['EC'][2,2]))*(args['EC'][3,3]/(args['wc2']+Ampc2+args['EC'][3,3])))**(-1/2)

    return(2*np.pi*(g_pulse-g_idle))



def coupling_correction_c23(t, args):
    Ampc2 = 0

    if 'Sigmoid' in args['Pulse_c2']:
        Ampc2 = args['Pulse_c2']['Amp'] * activation_pulse_int(t, args)
    elif 'Erfc' in args['Pulse_c2']:
        Ampc2 = args['Pulse_c2']['Amp'] * Pulse_2(t, args)


    g_idle = -0.5*args['EC'][4,3]*((args['EC'][4,4]/(args['w3']+args['EC'][4,4]))*(args['EC'][3,3]/(args['wc2']+args['EC'][3,3])))**(-1/2)
    g_pulse = -0.5*args['EC'][4,3]*((args['EC'][4,4]/(args['w3']+args['EC'][4,4]))*(args['EC'][3,3]/(args['wc2']+Ampc2+args['EC'][3,3])))**(-1/2)

    return(2*np.pi*(g_pulse-g_idle))

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

