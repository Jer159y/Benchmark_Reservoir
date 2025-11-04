using DifferentialEquations
using Makie, GLMakie, CairoMakie
using DynamicalSystems

using ReservoirComputing
using LinearAlgebra, Random

import ..ReservoirComputing: AbstractReservoirDriver, AbstractDriver, reservoir_driver_params, RNN, NonLinearAlgorithm, NLADefault
import ..ReservoirComputing: AbstractStates, StandardStates, AbstractPaddedStates
import ..ReservoirComputing: allocate_tmp, adapt, next_state!

include("function/generate_esn.jl")
include("function/change_initialstate.jl")