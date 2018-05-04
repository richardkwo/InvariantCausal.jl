using Distributions: cdf, FDist, TDist
using StatsBase: sample

"""
    testConditionalIndep(X, y, S, env, n_env:; α=0.01, method="chow")

Test H0: `y | X_S` is invariant across environments

* `X`: n x p matrix of covariates
* `y`: vector of response
* `env`: vector of environments (1:n_env), corresponding to rows of X
* `n_env`: number of environments
* `S`: vector of column indices of X
* `α`: significance level
* `method`: "chow" for chow test

Return: `rej`, `p_value`, `conf_intervals`
* `rej`: false if invariant; true if not invariant
* `conf_intervals`: |S| x 2 matrix, cols = (min, max)
"""
function testConditionalIndep(X::Matrix{Float64}, y::Vector{Float64}, env::Vector{Int64}, n_env::Int64, S::Vector{Int64};
                              α=0.01, method="chow")
    @assert issubset(S, 1:size(X, 2))
    @assert size(X, 1) == length(y) == length(env)
    if method == "chow"
        test_fun = conditional_indep_test_chow
    else
        error("method undefined")
    end
    return test_fun(X[:, S], y, env, n_env, α = α)
end

"""
    conditional_indep_test_chow(X, y, env, n_env; α=0.01)

Test H0: `y` | `X` is invariant under environments by combining leave-one-out chow test.

*`X`:                 n x p matrix of covariates
*`y`:                 vector of response
*`env`:               vector of environments (1:n_env), corresponding to rows of X
*`n_env`:             number of environments
*`α`:                 significance level
*`n_max_for_exact`:   maximum number of observations of an environment for exact testing; 
                      otherwise a subsample of n_max rows will be used

Return: `rej`, `p_value`, `conf_intervals`
* `rej`: false if invariant
* `conf_intervals`: p x 2 matrix, cols = (min, max)
"""
function conditional_indep_test_chow(X::Matrix{Float64}, y::Vector{Float64}, env::Vector{Int64}, n_env::Int64;
                                     α=0.01, n_max_for_exact=5000)
    p_values = zeros(n_env)
    p = size(X, 2)
    n_all = size(X, 1)
    for i in 1:n_env
        @assert sum(env.==i) > 0
        # fit on (-env), test on (env)
        idx_in = (1:n_all)[env.==i]
        idx_out = (1:n_all)[env.!=i]
        if length(idx_in) > n_max_for_exact
            idx_in = sample(idx_in, n_max_for_exact, replace=false)
        end
        p_values[i] = two_sample_chow(X[idx_out, :], X[idx_in, :], y[idx_out], y[idx_in];
                                      α=α, add_intercept=true)
        if p_values[i] < min(1e-6, α / n_env)
            # early stop to save time
            p_values[(i+1):end] = p_values[i]
            break
        end
    end
    # Bonferroni correction
    p_value = min(minimum(p_values) * n_env, 1.)
    if p_value < α
        # reject
        reject = true
        conf_intervals = zeros(p, 2)  # min = max = 0.
    else
        # not reject
        reject = false
        # pool all data and run regression
        X = hcat(X, ones(size(X,1)))
        β = (X' * X) \ (X' * y)
        σ = sqrt(sum((y - X * β).^2) / (n_all - p - 1))
        prec_X = diag(inv(X' * X))
        qt = quantile(TDist(n_all - p - 1), 1 - α / (2 * p))
        conf_left = β - qt * σ * sqrt.(prec_X)
        conf_right = β + qt * σ * sqrt.(prec_X)
        # note: the union of (1-α) conf ints have coverage (1-2α) (Theorem 2 of PBM)
        conf_intervals = hcat(conf_left[1:p], conf_right[1:p])
    end
    return reject, p_value, conf_intervals
end

"""
    two_sample_chow(X1, X2, y1, y2; [α=0.01, add_intercept=true])

Two-sample Chow's test of H0: two linear regressions `y1 ~ X1` and `y2 ~ X2` have 
the same linear coefficients, assuming Gaussian errors with equal variances.

Will fit on `X1`` and test residual on `X2`. 
Choose `X2` to be the sample with smaller n since n2 x n2 matrix inversion is performed.
"""
function two_sample_chow(X1::Matrix{Float64}, X2::Matrix{Float64},
                         y1::Vector{Float64}, y2::Vector{Float64}; α=0.01, add_intercept=true)
    n1 = size(X1, 1)
    n2 = size(X2, 1)
    if add_intercept
        X1 = hcat(X1, ones(n1))
        X2 = hcat(X2, ones(n2))
    end
    p = size(X1, 2)
    # fit on X1
    # _fit = glm(X1, y1, Normal(), IdentityLink())
    β = (X1' * X1) \ (X1' * y1)
    res2 = y2 - X2 * β
    Σ_res = diagm(ones(n2)) + X2 * ((X1' * X1) \ X2')   # inv(A) * B = A \ B
    σ2 = var(y1 - X1 * β) * (n1 - 1) / (n1 - p)  # should use dof = (n - p) as denominator
    chow_stat = res2' * (Σ_res \ res2) / (σ2 * n2)   # inv(A) * B = A \ B
    # F distribution
    ν1 = n2
    ν2 = n1 - size(X1, 2)
    p_value = 1 - cdf(FDist(ν1, ν2), chow_stat)
    return p_value
end