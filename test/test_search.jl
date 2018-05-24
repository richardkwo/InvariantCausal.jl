using Base.Test
using StatsBase.sample
using DataFrames

@time @testset "causal search" begin
    X = readdlm(joinpath(@__DIR__, "X1.dat"))
    env = Vector{Int}(X[:,1])
    X = X[:,2:end]
    S = 1:size(X,2)
    result = causalSearch(X, X[:, 1], env, setdiff(S,1) , α=0.01)
    @test result.model_reject == true
    result = causalSearch(X, X[:, 2], env, setdiff(S,2) , α=0.01, p_max=4)
    @test result.S == [5]
end

@time @testset "causal search logistic" begin
    df = DataFrame(x1 = CategoricalArray([1  0  0  0  0  1  0  1  1  0  0  0  0  1  0  1][:]),
                x2 = [0.0  0.0  1.0  0.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0  0.0  0.0  0.0  1.0  1.0][:],
                x3 = [0.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0][:],
                y =  [1.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0][:])
    env = repeat([1,2], inner=8)
    r1 = causalSearch(df, :y, env, method="logistic-LR", iterate_all=true)
    @test length(r1.S) == 0
    r2 = causalSearch(df, :y, env, method="logistic-SF", iterate_all=true)
    @test length(r2.S) == 0
end