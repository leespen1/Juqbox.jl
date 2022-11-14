println("Setup for ", eval_lab ? "lab frame evaluation" : "rotating frame optimization")

Ne = [Ne1, Ne2]
Ng = [Ng1, Ng2]

N = Ne1*Ne2; # Total number of nonpenalized energy levels
Ntot = (Ne1+Ng1)*(Ne2+Ng2)
Nguard = Ntot - N # total number of guard states

Nt1 = Ne1 + Ng1
Nt2 = Ne2 + Ng2

# construct the lowering and raising matricies: amat, bmat

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

# System Hamiltonian
if eval_lab
    H0 = 2*pi*( fa*N1 + fb*N2 -x1/2*(N1*N1-N1) - x2/2*(N2*N2-N2) - x12*(N1*N2) )
else
    H0 = 2*pi*( (fa-rot_freq[1])*N1 + (fb-rot_freq[2])*N2 -x1/2*(N1*N1-N1) - x2/2*(N2*N2-N2) - x12*(N1*N2) )
end
H0 = Array(H0)

# package the lowering and raising matrices together into an one-dimensional array of two-dimensional arrays
if eval_lab
    Hunc_ops=[Array(amat+adag), Array(bmat+bdag)]
else
    Hsym_ops=[Array(amat+adag), Array(bmat+bdag)]
    Hanti_ops=[Array(amat-adag), Array(bmat-bdag)]
end

maxpar = [amax, bmax]

# Estimate time step
if eval_lab
    Pmin = 100 # should be 20 or higher
    nsteps = calculate_timestep(Tmax, H0, Hunc_ops, maxpar, Pmin)
else
    Pmin = 40 # should be 20 or higher
    nsteps = calculate_timestep(Tmax, H0, Hsym_ops, Hanti_ops, maxpar, Pmin)
end

println("Number of time steps = ", nsteps)

om = zeros(Nctrl, Nfreq) # Allocate space for the carrier wave frequencies

@assert(Nfreq==1 || Nfreq==2 || Nfreq==3)
if Nfreq == 2
    # ctrl 1
    om[1,1] = 2*pi*(fa - rot_freq[1])
    om[1,2] = 2*pi*(fa - rot_freq[1] - x12) # coupling freq for both ctrl funcs (re/im)
    # ctrl 1
    om[2,1] = 2*pi*(fb - rot_freq[2])
    om[2,2] = 2*pi*(fb - rot_freq[2] - x12) # coupling freq for both ctrl funcs (re/im)
elseif Nfreq == 3
    om[1,2] = -2.0*pi*x1 # 1st ctrl, re
    om[2,2] = -2.0*pi*x2 # 2nd ctrl, re
    om[1:Nctrl,3] .= -2.0*pi*x12 # coupling freq for both ctrl funcs (re/im)
end
println("Carrier frequencies (lab frame) 1st ctrl Hamiltonian [GHz]: ", rot_freq[1] .+ om[1,:]./(2*pi))
println("Carrier frequencies (lab frame) 2nd ctrl Hamiltonian [GHz]: ", rot_freq[2] .+ om[2,:]./(2*pi))


# Initial basis with guard levels
U0 = initial_cond(Ne, Ng)

utarget = U0 * gate_cnot


# rotation matrices
omega1, omega2 = Juqbox.setup_rotmatrices(Ne, Ng, rot_freq)

# Compute Ra*Rb*utarget
rot1 = Diagonal(exp.(im*omega1*Tmax))
rot2 = Diagonal(exp.(im*omega2*Tmax))

if eval_lab
    vtarget = utarget # target in the lab frame
else    
    vtarget = rot1*rot2*utarget # target in the rotating frame
end

# assemble problem description for the optimization
if eval_lab
    params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                              Hconst=H0, Hunc_ops=Hunc_ops, use_sparse=use_sparse)
else
    params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                              Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, use_sparse=use_sparse)
end

# initial parameter guess
if eval_lab
    startFromScratch = false
else
    startFromScratch = true
end
startFile = "../../examples/drives/cnot2-pcof-opt-t50-avg.jld2"

# dimensions for the parameter vector
D1 = 10 # number of B-spline coeff per oscillator, freq and sin/cos

nCoeff = 2*Nctrl*Nfreq*D1 # Total number of parameters.

Random.seed!(2456)
if startFromScratch
    pcof0 = amax*0.01 * rand(nCoeff)
    println("*** Starting from pcof with random amplitude ", amax*0.01)
else
    # the data on the startfile must be consistent with the setup!
    # use if you want to have initial coefficients read from file
    pcof0 = read_pcof(startFile)
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Nctrl*Nfreq) # factor '2' is for sin/cos
    nCoeff = 2*Nctrl*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements
    println("*** Starting from B-spline coefficients in file: ", startFile)
end


# min and max B-spline coefficient values
minCoeff, maxCoeff = Juqbox.assign_thresholds(params,D1,maxpar)
println("Number of min coeff: ", length(minCoeff), "Max Coeff: ", length(maxCoeff))


println("*** Settings ***")
# output run information
println("Frequencies: fa = ", fa, " fb = ", fb, " fa-favg = ", fa-favg, " fb-favg = ", fb-favg )
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

estimate_Neumann!(new_tol, params, maxpar);
println("Using tolerance", new_tol, " and ", params.linear_solver.max_iter, " terms in the Neumann iteration")

# Allocate all working arrays
wa = Juqbox.Working_Arrays(params, nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=maxIter, lbfgsMax=lbfgsMax, startFromScratch=startFromScratch)

#uncomment to run the gradient checker for the initial pcof
#=
if @isdefined addOption
    addOption( prob, "derivative_test", "first-order"); # for testing the gradient
else
    AddIpoptStrOption( prob, "derivative_test", "first-order")
end
=#

#uncomment to change print level
#=
if @isdefined addOption
    addOption(prob, "print_level", 0); # for testing the gradient
else
    AddIpoptIntOption(prob, "print_level", 0)
end
=#

println("Initial coefficient vector stored in 'pcof0'")

if eval_lab
    objf, uhist, trfid = traceobjgrad(pcof0, params, wa, true, false); # evaluate objective function, but not the gradient
    println("Trace fidelity: ", trfid);
end
