import numpy as np
from juliacall import Main as jl
import convert_juqbox_pcof_to_IQ as pcof_tool # For converting pcof to a pulse
jl.seval("using Juqbox")
jl.seval('include("juqbox_pulse_gen.jl")')

# System parameters, these should be obtained automatically from some JSON library
parameters = {
    "fa" : 4.10336,
    "xa" : -0.2198
}

# Defaults, just to remember sensible values. Not used in function.
T = 100.0
D1 = 10

# Prepare the CNOT gate
cnot_gate = np.zeros((4,4))
cnot_gate[0,0] = 1.0
cnot_gate[1,1] = 1.0
cnot_gate[2,3] = 1.0
cnot_gate[3,2] = 1.0


"""
Changes to make:
- handle unitary input as list of lists, not numpy array (or both)
- fa and xa should be obtained automatically from JSON file, not input as global variable
- Make sure units are consistent between juqbox and TFOC
- Where are the carrier frequencies? Is that user input or calculated by Juqbox
  based on fa and xa?
- Eventually make 2-qubit, 3-qubit interface.

Differences from TFOC_pulse_gen input:

- D1 not in TFOC_pulse_gen, taking as keyword. Should use more descriptive name
  (# number of B-spline coeff per oscillator, freq and sin/cos)
- TFOC uses ampcut, not maxctrl. But I think they serve a similar purpose
- No weight argument. I don't think we have an equivalent 
- No bounds argument. I don't think we have an equivalent
- In Juqbox, sampling_rate does not affect the optimization calculations, only 
  the size of the pulse output (once the bspline coeffs are calculated, any
  sample rate could be used). I think "D1" fulfills a similar role (the number
  you give to determine how many parameters to optimize over)
- No initial_pulse argument. We could provide pcof0, or reverse
  engineer pcof from a pulse (Anders or Daniel would know more about the
  feasibility of that).
- We don't have any of the file saving or graphing. Might be easy to do this
  with TFOC/MEsolve once we have the pulse generated.
"""
def juqbox_pulse_gen(
    nstates, unitary, sampling_rate, pulse_length,
    D1=10, nguard=2, maxctrl=0.001*2*np.pi*8.5):

    ## Parameter list of TFOC_pulse_gen
    #(nstates, unitary, sampling_rate, pulse_length,
    #plot=True, save=False, fname='test', overwrite=True, mesolve=False, 
    #initial_pulse=[], weight=1, fine=False, ampcut=5, tol=1e-09, bounds=None)

    # Set up problem
    params, pcof0, maxAmp = jl.get_one_qubit_prob(
        nstates, nguard,
        parameters["fa"], parameters["xa"],  # Currently obtained
        maxctrl, D1,
        pulse_length, unitary)

    # Perform optimation
    pcof = jl.Juqbox.run_optimizer(params, pcof0, maxAmp)


    # Generate pulse using b-spline coefficients (pcof)
    pulse = pcof_tool.generate_pulse_from_pcof(T, D1, params.Ncoupled, params.Nunc, params.Cfreq, pcof, sampling_rate)

    p, q = pulse
    
    fidelity = 1.0 - params.last_infidelity # Check with Anders if this is correct

    return p, q, fidelity

# The "help" message for TFOC_pulse_gen (the original optimal control pulse generator)
"""
help(TFOC_pulse_gen)

Help on function TFOC_pulse_gen in module OC_pulse_generation:

TFOC_pulse_gen(nstates, unitary, sampling_rate, pulse_length, plot=True, save=False, fname='test', overwrite=True, mesolve=False, initial_pulse=[], weight=1, fine=False, ampcut=5, tol=1e-09, bounds=None)
    Enables optimal control pulse generation through Tensor Flow.

    Parameters
    ----------
    nstates (integer)
    target_unitary (list of lists)
    sampling rate (int, per ns) Used in the Tensor Flow calculation. Selecting too
       low a sampling rate may result in an inaccurate pulse.
    pulse_length (ns)

    optional arguments :
    plot (bool) Default: True
    save (list [bool, string]) Default: [False, 'test']. Will automatically append
       .dat to the string provided and save in the directory that the notebook
       was launched from.
    mesolve(bool) Default: False. If you turn it off, you will see 2x1 figure without
       the qutip solve of the time evolution of the calculated pulse
    initial_pulse(list) : input should be [p,q] as returned from TFOC routine. The unit needs to be in MHz. If empty, it uses a default initial amplitudes.
    weight (float) Defult: 1
        It give weight on loss functions. It should be between 0 and 1.
    ampcut (MHz) Default: 5. Chosen to stay below the hardware (OPX) voltage limit.
    bounds () Default: None. Passed directly to bounds parameter in scipy.optimize.minimize

    Returns
    -------
    p, q. (MHz)
    fidelity

Example Usage
"""
