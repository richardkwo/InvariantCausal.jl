using Base.Test
using StatsBase.sample

X = readdlm(joinpath(@__DIR__, "X1.dat"))
env = Vector{Int}(X[:,1])
X = X[:,2:end]
S = 1:size(X,2)

@time @testset "causal search" begin
    result = causalSearch(X, X[:, 1], env, setdiff(S,1) , α=0.01)
    @test result.model_reject == true
    result = causalSearch(X, X[:, 2], env, setdiff(S,2) , α=0.01, p_max=4)
    @test result.S == [5]
end