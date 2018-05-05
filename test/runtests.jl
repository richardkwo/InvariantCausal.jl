#!/usr/bin/env julia
Pkg.test("GLMNet")   # force it to compile the dep
using InvariantCausal
using Base.Test
println("Starting tests")
@time begin
include("test_regression.jl") 
include("test_search.jl")
end