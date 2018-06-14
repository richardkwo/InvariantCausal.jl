using Distributions: cdf, FDist, TDist, Chisq, Bernoulli
using StatsBase: sample
using DataFrames: DataFrame
using StatsModels: @formula, Formula
using GLM: glm, loglikelihood, coef, confint, predict

"""
    conditional_inv_test_chow(X, y, env, n_env; α=0.01)

Test H0: `y` | `X` is invariant under environments by combining leave-one-out chow test.

* `X`:                 n x p matrix of covariates
* `y`:                 vector of response
* `env`:               vector of environments (1:n_env), corresponding to rows of X
* `n_env`:             number of environments
* `α`:                 significance level
* `n_max_for_exact`:   maximum number of observations of an environment for exact testing; 
                      otherwise a subsample of n_max rows will be used

Return: `rej`, `p_value`, `conf_intervals`
* `rej`: false if invariant
* `conf_intervals`: p x 2 matrix, cols = (min, max)
"""
function conditional_inv_test_chow(X::Matrix{Float64}, y::Vector{Float64}, env::Vector{Int64}, n_env::Int64;
                                     α=0.01, n_max_for_exact=5000)
    @assert n_env >= 1
    p_values = ones(n_env)
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
        if p_values[i] < min(α / n_env)
            break   # early termination
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
    β = zeros(p)
    try
        β = (X1' * X1) \ (X1' * y1)
    catch _err
        print_with_color(:light_red, "encountered $_err in least square for Chow's test\n")
        return 0.
    end
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

"""
    conditional_inv_test_logistic(df, :target, S, env, n_env; α=0.01, add_intercept=true, method="logistic-LR")

Test `H0: y | X` is invariant under environments by combining leave-one-out likelihood ratio test.
The model is logistic regression specified by `fmla`.

* `df`:                a DataFrames.DataFrame
* `target`:            target variable in df (Symbol)
* `S`:                 covariates to condition on (`X = df[:,S]`)
* `env`:               vector of environments (1:n_env), corresponding to rows of X
* `n_env`:             number of environments
* `α`:                 significance level
* `add_intercept`:     add `+ 1` or not in formula
* `method`:            
    + `logistic-LR`: likelihood ratio test
    + `logistic-SF`: test equal mean and variance of prediction errors with Sukhatme-Fisher

Return: `rej`, `p_value`, `conf_intervals`
* `rej`: false if invariant
* `conf_intervals`: p x 2 matrix, cols = (min, max)
"""
function conditional_inv_test_logistic(df::DataFrame, target::Symbol, S::Vector{Symbol},
                                          env::Vector{Int64}, n_env::Int64; α=0.01, add_intercept=true, method="logistic-LR")
    @assert n_env >= 1
    @assert !(target in S)
    p = length(S) + add_intercept    
    n_all = size(df, 1)
    p_values = ones(n_env)    
    fmla = get_formula(df, target, S, add_intercept=add_intercept)
    # fit on pooled data
    fit0 = glm(fmla, df, Bernoulli())
    # iterate over environments
    for i in 1:n_env
        @assert sum(env.==i) > 0
        if method == "logistic-LR"
            # fit separately
            fit1 = glm(fmla, df[env.==i, :], Bernoulli())
            fit2 = glm(fmla, df[env.!=i, :], Bernoulli())
            # log likelihood ratio = 2 log (p(bigger model) / p(smaller model)) 
            lr = 2 * (loglikelihood(fit1) + loglikelihood(fit2) - loglikelihood(fit0))
            p_values[i] = 1 - cdf(Chisq(p), lr)
        elseif method == "logistic-SF"
            # predict and test equal mean of residuals
            if p > 1
                p_hat = predict(fit0, df)
            else
                p_hat = predict(fit0)
            end
            res = (df[target] - p_hat) ./ sqrt.(p_hat .* (1 - p_hat))
            p_values[i] = sukhatme_fisher_test(res[env.==i], res[env.!=i])
        else
            error("method undefined")
        end
        if p_values[i] < α / n_env
            break   # early termination
        end
    end
    # Bonferroni correction
    p_value = min(minimum(p_values) * n_env, 1.)
    if p_value < α
        # reject
        reject = true
        conf_intervals = zeros(length(S), 2)  # min = max = 0.
    else
        # not reject
        reject = false
        # the pooled fit is accepted
        if length(S) == 0
            conf_intervals = zeros(0, 2)
        elseif add_intercept
            conf_intervals = confint(fit0)[2:end, :]  # intercept is always the 1st row
        else
            conf_intervals = confint(fit0)
        end
    end
    return reject, p_value, conf_intervals
end

function conditional_inv_test_logistic_LR(df::DataFrame, target::Symbol, S::Vector{Int64},
                                            env::Vector{Int64}, n_env::Int64; α=0.01, add_intercept=true)
    conditional_inv_test_logistic_LR(df, target, names(df)[S], env, n_env; α=α, add_intercept=add_intercept)                                            
end

"""
    get_formula(df::DataFrame, target::Symbol, S=setdiff(names(df), [target]); add_intercept=true)

Generate formula of `y ~ .` type for `df`, where `y` is specified by `target`.
"""
function get_formula(df::DataFrame, target::Symbol, S=setdiff(names(df), [target]); add_intercept=true)
    if length(S)==0
        return @eval @formula($target ~ 1)
    end
    if add_intercept
        return @eval @formula($target ~ +(1, $(S...)))
    else
        return @eval @formula($target ~ +($(S...), 0))
    end
end

"""
    sukhatme_fisher_test(x, y)

Sukhatme-Fisher test of H0: vectors x and y are two independent normal samples with equal mean and variance

See Perng, S. K., and Ramon C. Littell. "A test of equality of two normal population means and variances."
    Journal of the American Statistical Association 71.356 (1976): 968-971.

Return: p-value
"""
function sukhatme_fisher_test(x::Vector{Float64}, y::Vector{Float64})
    m = length(x)
    n = length(y)
    ss1 = var(x) * (m - 1)
    ss2 = var(y) * (n - 1)
    T = (mean(y) - mean(x)) / sqrt((m + n) * (ss1 + ss2) / ((m + n - 2) * (m * n)))
    F = (ss2 * (m-1)) / (ss1 * (n-1))
    g = FDist(n-1, m-1)
    if F >= median(g)
        H = 2 * (1 - cdf(g, F))
    else
        H = 2 * cdf(g, F)
    end 
    W = - 2 * log(H)
    t = TDist(m + n - 2)
    w = Chisq(2)
    Q = - 2 * log(2 * (1 - cdf(t, abs(T)))) - 2 * log(1 - cdf(w, W))
    q = Chisq(4)
    return 1 - cdf(q, Q)
end