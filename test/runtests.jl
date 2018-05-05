#!/usr/bin/env julia
using GLMNet
using InvariantCausal
using Base.Test

println("Starting tests")
@time begin
include("test_regression.jl") 
include("test_search.jl")
end