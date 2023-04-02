![Control functions](examples/cnot2-pop.png)

# Juqbox.jl

Juqbox.jl is a package for solving quantum optimal control problems in closed quantum systems, where the evolution of the state vector is governed by Schroedinger's equation.

The main features of Juqbox include
- Symplectic time integration of Schroedinger's equation using the Stormer-Verlet scheme.
- Efficient parameterization of the control functions using B-splines with carrier waves.
- Objective function includes target gate infidelity and occupation of guarded (forbidden) states.
- Exact computation of the gradient of the objective function by solving the discrete adjoint equation.

The numerical methods in Juqbox.jl are documented in these papers:
1. N. A. Petersson and F. M. Garcia, "Optimal Control of Closed Quantum Systems via B-Splines with Carrier Waves", SIAM J. Sci. Comput. (2022) 44(6): A3592-A3616, LLNL-JRNL-823853, [arXiv:2106.14310](https://arxiv.org/abs/2106.14310).
2. N. A. Petersson, F. M. Garcia, A. E. Copeland, Y. L. Rydin and J. L. DuBois, “Discrete Adjoints for Accurate Numerical Optimization with Application to Quantum Control”, LLNL-JRNL-800457, [arXiv:2001.01013](https://arxiv.org/abs/2001.01013).

## Installation

The following instructions assume that you have already installed Julia (currently version 1.8.5) on your system. Before proceeding, we recommend that you add the following to the file ~/.julia/config/startup.jl. You may have to first create the config folder under .julia in your home directory. Then add this line to the startup.jl file:

- **ENV["PLOTS_DEFAULT_BACKEND"]="GR"**

This is an environment variable. It specifies the backend for plotting. Most of the examples in this document uses the GR backend, which assumes that you have installed that package. If you have trouble with GR, you can instead install the "PyPlot" package and set the default backend to "PyPlot".

Start julia and type `]` to enter the package manager. First install these packages (unless they are already installed):
- (@v1.8) pkg> add Plots
- (@v1.8) pkg> add GR
- (@v1.8) pkg> add FFTW
- (@v1.8) pkg> add Ipopt
- (@v1.8) pkg> add JLD2
- (@v1.8) pkg> add LaTeXStrings
- (@v1.8) pkg> add Printf
- (@v1.8) pkg> add Random
- (@v1.8) pkg> add SparseArrays
- (@v1.8) pkg> add Test

Then install Juqbox.jl:
- (@v1.8) pkg> add  https://github.com/LLNL/Juqbox.jl
- (@v1.8) pkg> precompile
- (@v1.8) pkg> test Juqbox
- ... all tests should pass (case=flux gives a Warning message) ...

To exit the package manager you type `<DEL>`, and to exit julia you type `exit()`.

## Documentation

The Juqbox.jl documentation can be found [here](https://software.llnl.gov/Juqbox.jl/).

## Examples

To access the examples, clone the Juqbox.jl git repository:
- shell> git clone https://github.com/LLNL/Juqbox.jl.git

Then go to the examples directory in the juqbox.jl folder:
- shell> de juqbox.jl/examples

Start julia and try the `cnot1-setup.jl' test case:
- shell> julia
- julia> include("cnot1-setup.jl")
- julia> pcof = run_optimizer(prob,pcof0);
- julia> pl = plot_results(params,pcof);
- julia> pl[1]

Examples of the setup procedure can be found in the following scripts in the **Juqbox.jl/examples** directory (invoke by, e.g. **include("cnot1-setup.jl")**) 
- **rabi-setup.jl** Pi-pulse (X-gate) for a qubit, i.e. a Rabi oscillator.
- **cnot1-setup.jl** CNOT gate for a single qudit with 4 essential and 2 guard levels. 
- **flux-setup.jl** CNOT gate for single qubit with a flux-tuning control Hamiltonian.
- **cnot2-setup.jl** CNOT gate for a pair of coupled qubits with guard levels.
- **cnot3-setup.jl** Cross-resonance CNOT gate for a pair of qubits that are coupled by a cavity resonator. **Note:** This case reads an optimized solution from file.
- **Risk_Neutral/run_all.jl** SWAP 0-2 gate for a single qudit. This routine performs both a deterministic optimization, and a risk-neutral optimization
where the system Hamiltonian is perturbed by additive noise which is assumed to be uniform. Full details of the example can be found in Section 6.2 in one of our papers, which can be found [here](https://arxiv.org/abs/2106.14310).

## Contributing to Juqbox.jl
Juqbox.jl is currently under development. The prefered method of contributing is through a pull request (PR). If you are interested in contributing, please contact Anders Petersson (petersson1@llnl.gov) or Fortino Garcia (fortino.garcia@colorado.edu).

## Credits
Most of the Julia code was written by Anders Petersson and Fortino Garcia. Important contributions were also made by Ylva Rydin and Austin Copeland. 

## License
Juqbox.jl is relased under the MIT license.

### Note: FFTW.jl license 
Juqbox.jl uses the Julia package FFTW.jl for post processing of the
results. That package is released under the MIT Expat license and provides Julia bindings to the
FFTW library for fast Fourier transforms (FFTs), as well as functionality useful for signal
processing. Note that the FFTW library is licensed under GPLv2 or higher (see its license file), but
the bindings to the FFTW library in the FFTW.jl package are licensed under MIT. As an lternative to
using the FFTW libary, the FFTs in Intel's Math Kernel Library (MKL) can be used by setting an
environment variable JULIA_FFTW_PROVIDER to MKL and running Pkg.build("FFTW"). MKL will be provided
through MKL_jll. Setting this environment variable only needs to be done for the first build of the
package; after that, the package will remember to use MKL when building and updating.

### Note: Ipopt.jl license 
Juqbox.jl uses the Julia package Ipopt.jl for optimizing control
functions. That package is released under the MIT Expat License and provides Julia bindings to the
Ipopt library, which is released under the Eclipse Public License.





