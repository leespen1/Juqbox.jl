import numpy as np
from juliacall import Main as jl

# Run CNOT2 setup on the Julia side
jl.include("../../examples/cnot2-setup.jl")
# Now the global variables/functions from that script can be accessed using the
# syntax `jl.VariableName`.

# Get optimized control vector. Notice that everything on the right hand side
# of `=` is from the Julia side, not Python
pcof = jl.Juqbox.run_optimizer(jl.prob, jl.pcof0)
# pcof lives in the Python side, but is of type <class 'juliacall.VectorValue'>,
# so it's type will be interpreted correctly when passed to a Julia function

# For example, we can pass it to `plot_results` and there will not be an error,
# although I have been unable to display Julia plots from the Python REPL
#pl = jl.Juqbox.plot_results(jl.params, pcof)

# On the other hand, we can convert the control vector to a numpy array, but we
# will get an exception when passing it to strongly-typed Julia functions
#np_pcof = np.array(pcof)
# The following line will throw an exception if uncommented, because np_pcof will
# be treated as type `PythonCall.PyArray{Float64, 1, true, true, Float64}` on 
# the Julia side
#pl = jl.Juqbox.plot_results(jl.params, np_pcof)

# But we can use Juqbox to get the unitaryhistory ourselves
#objv, unitaryhistory, fidelity = jl.Juqbox.traceobjgrad(pcof, jl.params, jl.wa, True, False);

# This also shows how the most basic datatypes (e.g. Ints, Floats, Bools, etc)
# can be passed to Julia functions, even strongly-typed functions, without issue.
