using Distributions
using Base.Test

@time @testset "linear reg chow test" begin
    p = 7
    n = 500
    n_env = 3
    env = repeat(1:n_env, inner=n)
    X = randn(n * n_env, p)
    beta = randn(p)
    y = X * beta - 1. + randn(n * n_env)
    rej, p_value, conf_intervals = InvariantCausal.conditional_inv_test_chow(X, y, env, n_env)
    @test rej == false
    y[1:n] = X[1:n, :] * (beta / 2) - 1. + randn(n)  # change environment 1
    rej, p_value, conf_intervals = InvariantCausal.conditional_inv_test_chow(X, y, env, n_env)
    @test rej == true
end

@time @testset "chow test" begin
    m = 100
    p_values = map(x -> (
        n = 300;
        p = 7;
        X1 = randn(2 * n, p);
        X2 = randn(n, p);
        beta = randn(p);
        y1 = X1 * beta + 1 + randn(2 * n);
        y2 = X2 * beta + 1 + randn(n);
        two_sample_chow(X1, X2, y1, y2)), 1:m)
    @test abs(mean(p_values .< 0.05) - 0.05) < 2 * sqrt(0.05 * 0.95 / m)
end

