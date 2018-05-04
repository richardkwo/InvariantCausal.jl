import Base.cov
using StatsBase: sample
using UnicodePlots.spy

abstract type SEM end

struct GaussianSEM <: SEM
    p       ::Int64
    B       ::Matrix{Float64}
    err_var ::Vector{Float64}
    function GaussianSEM(B, err_var)
        @assert size(B, 1) == size(B, 2) == length(err_var)
        @assert all(err_var .> 0)
        p = length(err_var)
        new(p, B, err_var)
    end
end

function Base.show(io::IO, sem::GaussianSEM)
    print(io, "Gaussian SEM with $(sem.p) variables:\n")
    print(io, "B = \n")
    print(io, spy(sem.B))
    print(io, "σ² = $(sem.err_var)")
end

"""
    simulate(sem, [n])
    simulate(sem, [do_variables, do_values], [n])

Simulate from a Gaussian SEM `sem`. `n` is the sample size.
do-interventions can be performed by specifying vectors of `do_variables` and `do_values`.
"""
function simulate(sem::GaussianSEM)
    p = sem.p
    ϵ = randn(p) .* sqrt.(sem.err_var)
    return (eye(p) - sem.B) \ ϵ
end

function simulate(sem::GaussianSEM, do_variables::Vector{Int64}, do_values::Vector{Float64})
    @assert length(do_variables) == length(do_values)
    p = sem.p
    ϵ = randn(p) .* sqrt.(sem.err_var)
    ϵ[do_variables] = do_values
    B = copy(sem.B)
    B[do_variables, :] = 0
    return (eye(p) - B) \ ϵ
end

function simulate(sem::GaussianSEM, n::Int64)
    return vcat(map(i -> simulate(sem), 1:n)'...)
end

function simulate(sem::GaussianSEM, do_variables::Vector{Int64}, do_values::Vector{Float64}, n::Int64)
    return vcat(map(i -> simulate(sem, do_variables, do_values), 1:n)'...)
end

function causes(sem::SEM, i::Int64)
    @assert 1 <= i <= sem.p
    return (1:sem.p)[sem.B[i, :].!=0]
end

function cov(sem::GaussianSEM)
    S = inv(eye(sem.p) - sem.B)
    return S * diagm(sem.err_var) * S'
end

"""
    random_gaussian_SEM(p, k; [lb=-2, ub=2, var_min=0.5, var_max=2])

Generate a random-graph acyclic SEM with `p` variables and `k` average degree, and random coefficients.
* `lb`, `ub`: coeff  ~ unif[`lb`, `ub`] with random sign
* `var_min`, `var_max`: var of error ~ unif[`var.min`, `var.max`]
"""
function random_gaussian_SEM(p::Int64, k::Int64; lb=-2, ub=2, var_min=0.5, var_max=2)
    B = zeros(p, p)
    B[rand(p, p) .< 2k / (p-1)] = 1
    B[UpperTriangular(B).!=0] = 0
    m = sum(B.==1)
    B[B.==1] = (rand(m) * (ub - lb) + lb) .* sign.(randn(m))
    err_var = rand(p) * (var_max - var_min) + var_min
    _order = sample(1:p, p, replace=false)
    B = B[_order, _order]
    return GaussianSEM(B, err_var)
end

"""
    random_noise_intervened_SEM(sem::GaussianSEM, [p_intervened=2, noise_multiplier_min=0.5, noise_multiplier_max=2., avoid=[],
                                                   prob_coeff_unchanged=2/3, lb=-2, ub=2])

Produce a new SEM based on original SEM by changing coefficients and noise variances.
* `p_intervened`: randomly choose `p_intervened` variables to intervene; will avoid those specified in `avoid`
* [`noise_multiplier_min`, `noise_multiplier_max`]: interval that noise multiplier is uniformly sampled from
* `prob.coeff.unchanged`: probability that coefficient is not changed
* `[lb, ub]`: if to change, coefficient is drawn uniformly from this interval with random sign

Return: `sem_new`, `intervened_variables`
"""
function random_noise_intervened_SEM(sem::GaussianSEM;
                                     p_intervened=2, noise_multiplier_min=0.5, noise_multiplier_max=2., avoid=[],
                                     prob_coeff_unchanged=2/3, lb=-2, ub=2)
    B = copy(sem.B)
    p = sem.p
    err_var = copy(sem.err_var)
    vars = sample(setdiff(collect(1:p), avoid), p_intervened, replace=false)
    for i in vars
        noise_multiplier = rand() * (noise_multiplier_max - noise_multiplier_min) + noise_multiplier_min
        err_var[i] = err_var[i] * noise_multiplier
        if rand() > prob_coeff_unchanged
            _J = (1:p)[B[i, :].!=0]
            B[i, _J] = rand(length(_J)) * (ub - lb) + lb
        end
    end
    return GaussianSEM(B, err_var), vars
end
