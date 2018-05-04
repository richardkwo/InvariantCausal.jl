#!/usr/bin/env julia
push!(LOAD_PATH, "../src/")
using GLM
using InvariantCausal
using Base.Test

println("Starting tests")
include("test_regression.jl") 
include("test_search.jl")