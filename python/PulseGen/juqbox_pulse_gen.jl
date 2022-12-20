using LinearAlgebra
using Plots
using FFTW
using DelimitedFiles
using Printf
using Ipopt
using Random

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox # quantum control module

using PythonCall
#==========================================================
This routine initializes an optimization problem to recover 
a CNOT gate on a single qudit with 4 energy levels (and 2 
guard states). The drift Hamiltonian in the rotating frame 
is
              H_0 = - 0.5*ξ_a(a^†a^†aa),
where a is the annihilation operator for the qudit. 
Here the control Hamiltonian includes the usual symmetric 
and anti-symmetric terms 
     H_{sym} = p(t)(a + a^†),    H_{asym} = q(t)(a - a^†)
which come from the rotating frame approximation and hence 
we refer to these as "coupled" controls.
The problem parameters are:
                ω_a =  2π × 4.10595   Grad/s,
                ξ_a =  2π × 2(0.1099) Grad/s.
We use Bsplines with carrier waves with frequencies
0, ξ_a, 2ξ_a Grad/s.
==========================================================# 
function get_one_qubit_prob(N::Int64, Nguard::Int64, fa::Float64,
        xa::Float64, maxctrl::Float64, D1::Int64, T::Float64, gate::AbstractMatrix)
    gate = pyconvert(Array{ComplexF64,2}, gate) # Convert to Julia array, to comply with type restrictions
    Ntot = N + Nguard # Total number of energy levels

    rot_freq = [fa]
    # form the Hamiltonian matrices
    H0, Hsym_ops, Hanti_ops = hamiltonians_one_sys(Ness=[N], Nguard=[Nguard], freq01=fa, anharm=xa, rot_freq=rot_freq)

    # calculate resonance frequencies
    om, Utrans = get_resonances(Ness=[N], Nguard=[Nguard], Hsys=H0, Hsym_ops=Hsym_ops)
    Nctrl = size(om, 1)
    Nfreq = size(om, 2)

    println("Nctrl = ", Nctrl, " Nfreq = ", Nfreq)

    # Estimate time step
    nsteps = calculate_timestep(T, H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, maxCop=[maxctrl], Pmin=40)
    println("Duration T = ", T, " # time steps: ", nsteps)

    # Initial basis with guard levels
    U0 = initial_cond([N], [Nguard])


    # Initial basis with guard levels
    U0 = initial_cond([N], [Nguard])
    utarget = U0 * gate # Add zero rows for the guard levels

    # create a linear solver object
    linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER,max_iter=100,tol=1e-12,nrhs=N)

    params = Juqbox.objparams([N], [Nguard], T, nsteps, Uinit=U0, Utarget=utarget, Cfreq=om, Rfreq=rot_freq, Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, linear_solver=linear_solver)

    # number of B-splines per ctrl/freq/real-imag
    nCoeff = 2*D1*Nctrl*Nfreq

    maxrand = 0.05*maxctrl/Nfreq  # amplitude of the random control vector
    pcof0 = init_control(params, maxrand=maxrand, nCoeff=nCoeff, seed=2345)

    # same ctrl threshold for all frequencies
    maxAmp = maxctrl/Nfreq .* ones(Nfreq)

    println("*** Settings ***")
    println("Number of coefficients per spline = ", D1, " Total number of control parameters = ", length(pcof0))
    println("Tikhonov coefficients: tik0 = ", params.tik0)
    println()
    println("Problem setup (Hamiltonian, carrier freq's, time-stepper, etc) is stored in 'params' object")
    println("Initial coefficient vector is stored in 'pcof0' vector")
    println("Max control amplitudes is stored in 'maxAmp' vector")

    return params, pcof0, maxAmp
end

