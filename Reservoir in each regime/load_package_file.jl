using DifferentialEquations
using Makie, GLMakie, CairoMakie, ColorSchemes
using DynamicalSystems, ChaosTools

using ReservoirComputing
using LinearAlgebra, Random, StaticArrays

import ..ReservoirComputing: AbstractReservoirDriver, AbstractDriver, reservoir_driver_params, RNN, NonLinearAlgorithm, NLADefault
import ..ReservoirComputing: AbstractStates, StandardStates, AbstractPaddedStates
import ..ReservoirComputing: allocate_tmp, adapt, next_state!

include("function/generate_esn.jl")
include("function/change_initialstate.jl")

using Infiltrator