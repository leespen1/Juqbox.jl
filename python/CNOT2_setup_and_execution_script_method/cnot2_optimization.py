import numpy as np
from juliacall import Main as jl

# Get problem parameters
jl.include("./cnot2_setup_parameters.jl")
jl.include("./cnot2_construct_problem.jl")
pcof = jl.Juqbox.run_optimizer(jl.prob, jl.pcof0)
