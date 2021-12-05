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
#pyplot()
using SparseArrays

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox

function initial_cond(Ntot, N, Ne, Ng)
    Ident = Matrix{Float64}(I, Ntot, Ntot)
    U0 = Ident[1:Ntot, 1:N] # Rectangular subset of identity
    # adjust the initial guess
    if Ng[1] + Ng[2] > 0
        Nt = Ne + Ng
        # build up a basis for the essential states
        col = 0
        m = 0
        for k2 in 1:Nt[2]
            for k1 in 1:Nt[1]
                m += 1
                # is this a guard level?
                guard = (k1 > Ne[1]) || (k2 > Ne[2])
                if !guard
                    col += 1
                    U0[:,col] = Ident[:,m]
                end # if ! guard
            end # for
        end # for
    end # if
    return U0
end

println("***This case ONLY evaluates the accuracy of a control pulse in the laboratory frame***")
Nosc = 2 # number of coupled oscillators

Ne1 = 2 # essential energy levels per oscillator 
Ne2 = 2
Ng1 = 1 # 0 # Osc-1, number of guard states
Ng2 = 1 # 0 # Osc-2, number of guard states

Ne = [Ne1, Ne2]
Ng = [Ng1, Ng2]

N = Ne1*Ne2; # Total number of nonpenalized energy levels
Ntot = (Ne1+Ng1)*(Ne2+Ng2)
Nguard = Ntot - N # total number of guard states

Nt1 = Ne1 + Ng1
Nt2 = Ne2 + Ng2

Tmax = 50.0 # 100.0 # Duration of gate

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10595    # official
fb = 4.81526   # official
rot_freq = [fa, fb] # rotational frequencies
x1 = 2* 0.1099  # official
x2 = 2* 0.1126   # official
x12 = 0.1 # Artificially large to allow coupling. Actual value: 1e-6 
  
# construct the lowering and raising matricies: amat, bmat, cmat
# and the system Hamiltonian: H0
#

# Note: The ket psi = ji> = e_j kron e_i.
# We order the elements in the vector psi such that i varies the fastest with i in [1,Nt1] and j in [1,Nt2]
# The matrix amat = I kron a1 acts on alpha in psi = beta kron alpha
# The matrix bmat = a2 kron I acts on beta in psi = beta kron alpha
a1 = Array(Bidiagonal(zeros(Nt1),sqrt.(collect(1:Nt1-1)),:U))
a2 = Array(Bidiagonal(zeros(Nt2),sqrt.(collect(1:Nt2-1)),:U))

I1 = Array{Float64, 2}(I, Nt1, Nt1)
I2 = Array{Float64, 2}(I, Nt2, Nt2)

# create the a, a^\dag, b and b^\dag vectors
amat = kron(I2, a1)
bmat = kron(a2, I1)

adag = Array(transpose(amat))
bdag = Array(transpose(bmat))

# number ops
num1 = Diagonal(collect(0:Nt1-1))
num2 = Diagonal(collect(0:Nt2-1))

# number operators
N1 = Diagonal(kron(I2, num1) )
N2 = Diagonal(kron(num2, I1) )

# System Hamiltonian in the lab frame
H0 = 2*pi*( fa*N1 + fb*N2 -x1/2*(N1*N1-N1) - x2/2*(N2*N2-N2) - x12*(N1*N2) )

# dense matrices run faster, but take more memory
Hunc_ops=[Array(amat+adag), Array(bmat+bdag)]
H0 = Array(H0)

# max coefficients, rotating frame
amax = 0.014 # max amplitude ctrl func for Hamiltonian #1
bmax = 0.020 # max amplitude ctrl func for Hamiltonian #2
maxpar = [amax, bmax]

# Estimate time step
Pmin = 200 # should be 20 or higher
nsteps = calculate_timestep(Tmax, H0, Hunc_ops, maxpar, Pmin)

println("Number of time steps = ", nsteps)

# package the lowering and raising matrices together into an one-dimensional array of two-dimensional arrays
# Here we choose dense or sparse representation
#use_sparse = true
use_sparse = false

Nfreq = 2 # number of carrier frequencies

om = zeros(Nosc, Nfreq) # Allocate space for the carrier wave frequencies

@assert(Nfreq==1 || Nfreq==2 || Nfreq==3)
if Nfreq == 2
    om[1:Nosc,2] .= -2.0*pi*x12 # coupling freq for both ctrl funcs (re/im)
elseif Nfreq == 3
    om[1,2] = -2.0*pi*x1 # 1st ctrl, re
    om[2,2] = -2.0*pi*x2 # 2nd ctrl, re
    om[1:Nosc,3] .= -2.0*pi*x12 # coupling freq for both ctrl funcs (re/im)
end
println("Carrier frequencies 1st ctrl Hamiltonian [GHz]: ", om[1,:]./(2*pi))
println("Carrier frequencies 2nd ctrl Hamiltonian [GHz]: ", om[2,:]./(2*pi))

# specify target gate
# target for CNOT gate N=2, Ng = 1 coupled
utarget = zeros(ComplexF64, Ntot, N)
@assert(Ng1 == 0 || Ng1 == 1 || Ng1 == 2)
if Ng1 == 0
    utarget[1,1] = 1.0
    utarget[2,2] = 1.0
    utarget[3,4] = 1.0
    utarget[4,3] = 1.0
elseif Ng1 == 1
    utarget[1,1] = 1.0
    utarget[2,2] = 1.0
    utarget[4,4] = 1.0
    utarget[5,3] = 1.0
elseif Ng1 == 2
    utarget[1,1] = 1.0
    utarget[2,2] = 1.0
    utarget[5,4] = 1.0
    utarget[6,3] = 1.0
end

# rotation matrices
omega1, omega2 = Juqbox.setup_rotmatrices(Ne, Ng, rot_freq)

# Compute Ra*Rb*utarget
rot1 = Diagonal(exp.(im*omega1*Tmax))
rot2 = Diagonal(exp.(im*omega2*Tmax))

# target in the lab frame
vtarget = utarget

U0 = initial_cond(Ntot, N, Ne, Ng)

# assemble problem description for the optimization
params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                          Hconst=H0, Hunc_ops=Hunc_ops)

# initial parameter guess
startFromScratch = false # true
startFile = "drives/cnot2-pcof-opt-t50.jld2"

# dimensions for the parameter vector
D1 = 10 # number of B-spline coeff per oscillator, freq and sin/cos

nCoeff = 2*Nosc*Nfreq*D1 # Total number of parameters.

Random.seed!(2456)
if startFromScratch
  pcof0 = amax*0.01 * rand(nCoeff)
  println("*** Starting from pcof with random amplitude ", amax*0.01)
else
    # the data on the startfile must be consistent with the setup!
    # use if you want to have initial coefficients read from file
    pcof0 = read_pcof(startFile)
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Nosc*Nfreq) # factor '2' is for sin/cos
    nCoeff = 2*Nosc*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements
    println("*** Starting from B-spline coefficients in file: ", startFile)
end

samplerate = 32 # for plotting
casename = "cnot2" # for constructing file names

# min and max B-spline coefficient values
useBarrier = true
minCoeff, maxCoeff = Juqbox.assign_thresholds(params,D1,maxpar)
println("Number of min coeff: ", length(minCoeff), "Max Coeff: ", length(maxCoeff))

maxIter = 150 # 0 #250 #50 # optional argument
lbfgsMax = 250 # optional argument

println("*** Settings ***")
# output run information
println("Frequencies: fa = ", fa, " fb = ", fb)
println("Coefficients in the Hamiltonian: x1 = ", x1, " x2 = ", x2, " x12 = ", x12)
println("Essential states in osc = ", [Ne1, Ne2], " Guard states in osc = ", [Ng1, Ng2])
println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
println("Number of B-spline parameters per spline = ", D1, " Total number of parameters = ", nCoeff)
println("Max parameter amplitudes: maxpar = ", maxpar)
println("Tikhonov regularization tik0 (L2) = ", params.tik0)
if use_sparse
    println("Using a sparse representation of the Hamiltonian matrices")
else
    println("Using a dense representation of the Hamiltonian matrices")
end

# Allocate all working arrays
wa = Juqbox.Working_Arrays(params, nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=maxIter, lbfgsMax=lbfgsMax, startFromScratch=startFromScratch)

# uncomment to run the gradient checker for the initial pcof
# addOption( prob, "derivative_test", "first-order"); # for testing the gradient
# addOption(prob, "print_level", 0); # for testing the gradient

println("Initial coefficient vector stored in 'pcof0'")

# evaluate objective function
objf, uhist, trfid = traceobjgrad(pcof0, params, wa, true, false);
println("Trace fidelity: ", trfid);

