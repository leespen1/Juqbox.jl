import numpy as np
import scipy
import math

# Data structure to hold parameters for the B-spline
class bcparams:
    def __init__(self):
        self.T = 0.0 # Duration of spline function
        self.D1 = 0  # Number of B-spline coefficients per control function
        self.om = 0     # Array to save carrier wave frequencies [rad/s], size Nfreq
        self.tcenter = 0
        self.dtknot = 0
        self.pcof = 0 # coefficients for all 2*Ncoupled splines, size Ncoupled*D1*Nfreq*2 (*2 because of sin/cos)
        self.Nfreq = 0# Number of frequencies
        self.Ncoeff = 0 # Total number of coefficients
        self.Ncoupled = 0 # Number of B-splines functions for the coupled ctrl Hamiltonians
        self.Nunc = 0# Number of B-spline functions  for the UNcoupled ctrl Hamiltonians
        
def initialize_bcparams(T, D1, Ncoupled, Nunc, omega, pcof):
    my_params = bcparams()
    my_params.T = T
    my_params.D1 = D1
    my_params.Ncoupled = Ncoupled
    my_params.Nunc = Nunc
    my_params.om = omega

    my_params.dtknot = T/(D1 -2)
    my_params.tcenter = np.zeros(D1)
    for i in range(1,D1+1):
        my_params.tcenter[i-1] = my_params.dtknot*(i-1.5)
        
    my_params.Nfreq = np.shape(omega)[1]
    my_params.Ncoeff = my_params.Nfreq*D1*2*(Ncoupled + Nunc)
    my_params.pcof = pcof
    
    return my_params

# Define the B-splines with the carrier wave
def bcarrier2(t, params, func):
    # for a single oscillator, func=0 corresponds to p(t) and func=1 to q(t)
    # in general, 0 <= func < 2*Ncoupled + Nunc

    # compute basic offset: func 0 and 1 use the same spline coefficients, but combined in a different way
    osc = func//2 # osc is base 0; 0<= osc < Ncoupled
    q_func = func%2 # q_func = 0 for p and q_func=1 for q
    
    f = 0.0 # initialize
    
    dtknot = params.dtknot
    width = 3*dtknot
    
    # -1 here is due to the fact that python is 0 based and julia is 1 based
    k = max(3, math.ceil(t/dtknot + 2))-1 # pick out the index of the last basis function corresponding to t
    k = min(k, params.D1-1) #  Make sure we don't access outside the array
    #print(k)
    if func < 2*(params.Ncoupled + params.Nunc):
        # Coupled and uncoupled controls
        for freq in range(0,params.Nfreq):
            fbs1 = 0.0 # initialize
            fbs2 = 0.0 # initialize
            # offset in parameter array (osc = 0,1,2,...
            # Vary freq first, then osc
            offset1 = 2*osc*params.Nfreq*params.D1 + freq*2*params.D1
            offset2 = 2*osc*params.Nfreq*params.D1 + freq*2*params.D1 + params.D1

            # 1st segment of nurb k, julia
            tc = params.tcenter[k]
            tau = (t - tc)/width
            fbs1 += params.pcof[offset1+k] * (9/8 + 4.5*tau + 4.5*tau**2)
            fbs2 += params.pcof[offset2+k] * (9/8 + 4.5*tau + 4.5*tau**2)
            
            # 2nd segment of nurb k-1
            tc = params.tcenter[k-1]
            tau = (t - tc)/width
            fbs1 += params.pcof[offset1+k-1] * (0.75 - 9 *tau**2)
            fbs2 += params.pcof[offset2+k-1] * (0.75 - 9 *tau**2)
            
            # 3rd segment of nurb k-2
            tc = params.tcenter[k-2]
            tau = (t - tc)/width
            fbs1 += params.pcof[offset1+k-2] * (9/8 - 4.5*tau + 4.5*tau**2)
            fbs2 += params.pcof[offset2+k-2] * (9/8 - 4.5*tau + 4.5*tau**2)
            
            #print(fbs1,' ',fbs2)
            # for carrier phase
            # p(t)
            if (q_func==1):
                f += fbs1 * math.sin(params.om[osc,freq]*t) + fbs2 * math.cos(params.om[osc,freq]*t) # q-func
            else:
                f += fbs1 * math.cos(params.om[osc,freq]*t) - fbs2 * math.sin(params.om[osc,freq]*t) # p-func
                
    return f

def generate_pulse_from_pcof(T,D1,Ncoupled,Nunc,Cfreq,pcof,samplerate):
    nplot = round(T*samplerate)
    td = np.linspace(0,T, num=nplot+1)
    td_len = len(td)

    # initialize the parameters for the B-spline
    splinepar = initialize_bcparams(T, D1, Ncoupled, Nunc, Cfreq, pcof);

    # compute p,q
    p2 = np.zeros(td_len)
    q2 = np.zeros(td_len)
    #print(bcarrier2(td[10],splinepar,0),' ',p_res[10])
    for i in range(0,td_len):
        p2[i] = bcarrier2(td[i],splinepar,0)
        q2[i] = bcarrier2(td[i],splinepar,1)

    # convert the unit to desired ones (Radians to Hz)
    I_function = p2/(2*math.pi)
    Q_function = q2/(2*math.pi)
    
    return I_function,Q_function
