#!/usr/bin/env julia
using InvariantCausal
using Test
println("Starting tests")
@time begin
include("test_regression.jl") 
include("test_search.jl")
end