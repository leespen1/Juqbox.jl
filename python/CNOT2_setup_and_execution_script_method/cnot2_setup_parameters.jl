#==========================================================
This routine initializes an optimization problem to recover 
a CNOT gate on a coupled 2-qubit system with 2 energy 
levels on each oscillator (with 1 guard state on one and 
2 guard states on the other). The drift Hamiltonian in 
the rotating frame is
    H_0 = - 0.5*ξ_a(a^†a^†aa) 
          - 0.5*ξ_b(b^†b^†bb) 
          - ξ_{ab}(a^†ab^†b),
where a,b are the annihilation operators for each qubit.
Here the control Hamiltonian in the rotating frame
includes the usual symmetric and anti-symmetric terms 
H_{sym,1} = p_1(t)(a + a^†),  H_{asym,1} = q_1(t)(a - a^†),
H_{sym,2} = p_2(t)(b + b^†),  H_{asym,2} = q_2(t)(b - b^†).
The problem parameters for this example are,
            ω_a    =  2π × 4.10595   Grad/s,
            ξ_a    =  2π × 2(0.1099) Grad/s,
            ω_b    =  2π × 4.81526   Grad/s,
            ξ_b    =  2π × 2(0.1126) Grad/s,
            ξ_{ab} =  2π × 0.1       Grad/s,
We use Bsplines with carrier waves with frequencies
0, ξ_a, 2ξ_a Grad/s for each oscillator.
==========================================================# 
using LinearAlgebra
using Ipopt
using Base.Threads
using Random
using DelimitedFiles
using Printf
using FFTW
using Plots
using SparseArrays

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox # quantum control module

eval_lab = false # true

Nctrl = 2 # Number of control Hamiltonians

Ne1 = 2 # essential energy levels per oscillator 
Ne2 = 2
Ng1 = 2 # 0 # Osc-1, number of guard states
Ng2 = 2 # 0 # Osc-2, number of guard states

N = Ne1*Ne2; # Total number of nonpenalized energy levels

Tmax = 50.0 # Duration of gate

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10595    # official
fb = 4.81526   # official
favg = 0.5*(fa+fb)
rot_freq = [fa, fb] # rotational frequencies
#rot_freq = [favg, favg] # rotational frequencies
x1 = 2* 0.1099  # official
x2 = 2* 0.1126   # official
x12 = 0.1 # Artificially large to allow fast coupling. Actual value: 1e-6 
  
# max coefficients, rotating frame
amax = 0.040 # 0.014 # max amplitude ctrl func for Hamiltonian #1
bmax = 0.040 # 0.020 # max amplitude ctrl func for Hamiltonian #2

# Here we choose dense or sparse representation
use_sparse = true
# use_sparse = false

Nfreq = 2 # number of carrier frequencies


# CNOT target for the essential levels
gate_cnot =  zeros(ComplexF64, N, N)
gate_cnot[1,1] = 1.0
gate_cnot[2,2] = 1.0
gate_cnot[3,4] = 1.0
gate_cnot[4,3] = 1.0



samplerate = 32 # for plotting
casename = "cnot2" # for constructing file names

maxIter = 50 # 0 #250 #50 # optional argument
lbfgsMax = 250 # optional argument

new_tol = 1e-12
