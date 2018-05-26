using GLMNet: glmnet

"""
    screen_lasso(X, y, pmax)

Screening out `pmax` variables by lasso (glmnet) on `X` and `y`.
"""
function screen_lasso(X::Matrix{Float64}, y::Vector{Float64}, pmax::Int64)
    fit = glmnet(X, y)
    p = size(X, 2)
    @assert pmax <= p
    betas = fit.betas
    n_vars = sum(betas.!=0, 1)[:]
    S = Set{Int64}()
    for n in unique(n_vars)
        Z = betas[:, n_vars.==n]
        non_zeros = sum(Z.!=0, 2)[:]
        vars = (1:p)[non_zeros .> 0]
        new_vars = setdiff(vars, S)
        Z = Z[new_vars, :]
        if length(S) + length(new_vars) > pmax
            # must break ties
            z = abs.(Z[:, end])
            idx = sort(1:length(z), by = i -> z[i], rev = true)
            new_vars =  new_vars[idx[1:(pmax - length(S))]]
        end
        union!(S, new_vars)
        length(S) == pmax && break
    end
    return sort(collect(S))
end

"""
    screen_HOLP(X, y, pmax)

Screen out `pmax` variables with HOLP projection. 

See: Wang, Xiangyu, and Chenlei Leng. "High dimensional ordinary least squares projection for screening variables." 
     Journal of the Royal Statistical Society: Series B (Statistical Methodology) 78.3 (2016): 589-611.

"""
function screen_HOLP(X::Matrix{Float64}, y::Vector{Float64}, pmax::Int64)
    n, p = size(X)
    @assert p >= n
    @assert pmax < p
    _X = X .- mean(X, 1)
    _X = _X ./ std(_X, 1)
    _y = y - mean(y)
    β = _X' * ((_X * _X' + 10 * eye(n)) \ _y)
    return sortperm(abs.(β), rev=true)[1:pmax]
end