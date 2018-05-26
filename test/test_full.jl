using InvariantCausal
using Base.Test
using StatsBase.sample

include(joinpath(@__DIR__, "test_search.jl"))
include(joinpath(@__DIR__, "test_regression.jl"))

X = readdlm(joinpath(@__DIR__, "X1.dat"))
env = Vector{Int}(X[:,1])
X = X[:,2:end]
S = 1:size(X,2)

@time @testset "causal search" begin
    result = map(i -> causalSearch(X, X[:, i], env, setdiff(S,i) , α=0.01), S)
    @test result[2].S == [5]
    @test result[3].S == [5]
    @test result[7].S == [4] || result[7].S == []
    for i in [1, 4, 5, 6]
        @test result[i].model_reject == true
    end
end

@time @testset "causal search with selection_only" begin
    result = map(i -> causalSearch(X, X[:, i], env, setdiff(S,i), α=0.01, selection_only=true), S)
    @test result[2].S == [5]
    @test result[3].S == [5]
    @test result[7].S == [4] || result[7].S == []
    for i in [1, 4, 5, 6]
        @test result[i].model_reject == true
    end
end

@time @testset "causal search with limited # of true causes" begin
    result = map(i -> causalSearch(X, X[:, i], env, setdiff(S,i), α=0.01, max_num_true_causes=3), S)
    @test result[2].S == [5]
    @test result[3].S == [5]
    @test result[7].S == [4] || result[7].S == []
    for i in [1, 4, 5, 6]
        @test result[i].model_reject == true
    end
end

@time @testset "causal search with subsampling" begin
    result = map(i -> causalSearch(X, X[:, i], env, setdiff(S,i), α=0.01, n_max_for_exact=100), S)
    @test result[2].S == [5]
    @test result[3].S == [5]
    @test result[7].S == [4] || result[7].S == []
    for i in [1, 4, 5, 6]
        @test result[i].model_reject == true
    end
end

function generate_setting(setting_configs; n_environments=2)
    n_int = sample(setting_configs["n_int"])
    n_obs = sample(setting_configs["n_obs"])
    p = sample(setting_configs["p"])
    k = sample(setting_configs["k"])
    lb_obs = sample(setting_configs["lb_obs"])
    ub_obs = lb_obs + sample(setting_configs["ub_lb_delta_obs"])
    _var_1 = sample(setting_configs["err_var_min"])
    _var_2 = sample(setting_configs["err_var_min"])
    err_var_min = min(_var_1, _var_2)
    err_var_max = max(_var_1, _var_2)
    noise_multiplier_min = sample(setting_configs["noise_multiplier_min"])
    if rand() < setting_configs["prob_fixed_noise_multiplier"]
        noise_multiplier_max = noise_multiplier_min
    else
        noise_multiplier_max = noise_multiplier_min + sample(setting_configs["noise_multiplier_max_min_delta"])
    end
    _l = sample(setting_configs["lb_int"])
    _u = sample(setting_configs["ub_int"])
    lb_int = min(_l, _u)
    ub_int = max(_l, _u)
    if rand() < setting_configs["prob_int_single"]
        n_int_variables = 1
        frac_int_variables = 1 / p
    else
        frac_int_variables = sample(setting_configs["frac_int_variables"])
        n_int_variables = floor(Int64, p * frac_int_variables)
    end
    prob_coefficient_unchanged = setting_configs["prob_coefficient_unchanged"]
    target = sample(1:p)
    SEMs = Vector{GaussianSEM}(n_environments)
    env = repeat([1], inner=n_obs)
    intervened_variables = Vector{Vector{Int64}}(n_environments)
    intervened_variables[1] = []
    SEMs[1] = random_gaussian_SEM(p, k, 
              lb=lb_obs, ub=ub_obs, var_min=err_var_min, var_max=err_var_max)
    true_β = SEMs[1].B[target, setdiff(1:p, target)]
    true_causes = (1:(p-1))[true_β .!= 0]
    for j in 2:n_environments
        SEMs[j], intervened_variables[j] = random_noise_intervened_SEM(SEMs[1], p_intervened=n_int_variables, 
                noise_multiplier_min=noise_multiplier_min, noise_multiplier_max=noise_multiplier_max, 
                avoid=[target], prob_coeff_unchanged=prob_coefficient_unchanged,
                lb=lb_int, ub=ub_int)
        env = vcat(env, repeat([j], inner=n_int))
    end
    return Dict(
        "n_environments" => n_environments, 
        "env" => env,
        "SEMs" => SEMs,
        "intervened_variables" => intervened_variables,
        "target" => target,
        "true_causes" => true_causes,
        "true_β" => true_β,
        "n_int" => n_int,
        "n_obs" => n_obs,
        "p" => p,
        "k" => k,
        "lb_obs" => lb_obs,
        "ub_obs" => ub_obs,
        "err_var_min" => err_var_min,
        "err_var_max" => err_var_max,
        "noise_multiplier_min" => noise_multiplier_min,
        "noise_multiplier_max" => noise_multiplier_max,
        "lb_int" => lb_int,
        "ub_int" => ub_int,
        "prob_coefficient_unchanged" => prob_coefficient_unchanged,
        "n_int_variables" => n_int_variables,
        "frac_int_variables" => frac_int_variables
    )
end

@time @testset "random instances" begin
    setting_configs = Dict(
    "n_int" => [100, 200, 300, 400, 500],
    "n_obs" => [100, 200, 300, 400, 500],
    "p" =>  collect(5:9),
    "k" => collect(1:4),
    "lb_obs" => collect(0.1:0.1:2),
    "ub_lb_delta_obs" => collect(0.1:0.1:1),
    "err_var_min" => collect(0.1:0.1:2),
    "err_var_max" => collect(0.1:0.1:2),
    "noise_multiplier_min" => collect(0.1:0.1:4),
    "noise_multiplier_max_min_delta" => collect(0.1:0.1:2),
    "lb_int" => collect(0.1:0.1:2),
    "ub_int" => collect(0.1:0.1:2),
    "prob_fixed_noise_multiplier" => 1/3,
    "prob_coefficient_unchanged" => 2/3,
    "prob_int_single" => 1/6,
    "frac_int_variables" => 1. ./ collect(1.1:0.1:3)
    )
    for trial in 1:10
        for n_environments in [2, 3, 5]
            setting = generate_setting(setting_configs, n_environments=n_environments)
            env = setting["env"]
            p = setting["p"]
            target = setting["target"]
            true_β = setting["true_β"]
            true_causes = setting["true_causes"]
            intervened_variables = setting["intervened_variables"]
            SEM_obs = setting["SEMs"][1]
            X = simulate(SEM_obs, setting["n_obs"])
            for j in 2:n_environments
                _X = simulate(setting["SEMs"][j], setting["n_int"])
                X = vcat(X, _X)
            end
            y = X[:, target]
            X = X[:, setdiff(1:p, target)]
            result = causalSearch(X, y, env, verbose=false)
            println("S = $(result.S), truth = $true_causes")                        
            @test issubset(result.S, true_causes)
            for k in result.S
                @test result.confint[k, 1] < true_β[k] < result.confint[k, 2]
            end
        end
    end
end

@time @testset "random instances, many variables with screening" begin
    setting_configs = Dict(
    "n_int" => [100, 200, 300, 400, 500],
    "n_obs" => [100, 200, 300, 400, 500],
    "p" =>  collect(20:50),
    "k" => collect(1:3),
    "lb_obs" => collect(0.1:0.1:2),
    "ub_lb_delta_obs" => collect(0.1:0.1:1),
    "err_var_min" => collect(0.1:0.1:2),
    "err_var_max" => collect(0.1:0.1:2),
    "noise_multiplier_min" => collect(0.1:0.1:4),
    "noise_multiplier_max_min_delta" => collect(0.1:0.1:2),
    "lb_int" => collect(0.1:0.1:2),
    "ub_int" => collect(0.1:0.1:2),
    "prob_fixed_noise_multiplier" => 1/3,
    "prob_coefficient_unchanged" => 2/3,
    "prob_int_single" => 1/6,
    "frac_int_variables" => 1. ./ collect(1.1:0.1:3)
    )
    for trial in 1:10
        for n_environments in [2, 3, 5]
            setting = generate_setting(setting_configs, n_environments=n_environments)
            env = setting["env"]
            p = setting["p"]
            target = setting["target"]
            true_β = setting["true_β"]
            true_causes = setting["true_causes"]
            intervened_variables = setting["intervened_variables"]
            SEM_obs = setting["SEMs"][1]
            X = simulate(SEM_obs, setting["n_obs"])
            for j in 2:n_environments
                _X = simulate(setting["SEMs"][j], setting["n_int"])
                X = vcat(X, _X)
            end
            y = X[:, target]
            X = X[:, setdiff(1:p, target)]
            result = causalSearch(X, y, env, verbose=false, p_max=8)
            # performance only guaranteed if lasso selected all the causes
            println("S = $(result.S), truth = $true_causes")            
            if issubset(true_causes, result.variables_considered)
                @test issubset(result.S, true_causes)
            end
        end
    end
end