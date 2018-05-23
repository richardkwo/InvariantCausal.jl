using DataStructures: PriorityQueue, enqueue!, dequeue_pair!
using DataFrames: DataFrame

struct CausalSearchResult
    S                               ::Union{Vector{Int64}, Vector{Symbol}}
    confint                         ::Matrix{Float64}
    α                               ::Float64
    p                               ::Int64
    variables_considered            ::Union{Vector{Int64}, Vector{Symbol}}
    selection_only                  ::Bool
    model_reject                    ::Bool
    function CausalSearchResult(S, confint, α, p, variables_considered; selection_only=false)
        new(collect(S), confint, α, p, variables_considered, selection_only, false)
    end
    function CausalSearchResult(S, α, p, variables_considered; selection_only=true)
        confint = Matrix{Float64}(length(S), 2)
        confint[:, 1] = -Inf
        confint[:, 2] = Inf
        new(collect(S), confint, α, p, variables_considered, selection_only, false)
    end
    function CausalSearchResult(α, p, variables_considered; selection_only=true)
        new(Vector{Int64}(), Matrix{Float64}(0, 2), α, p, variables_considered, selection_only, true)
    end
end

function Base.show(io::IO, result::CausalSearchResult)
    if result.model_reject
        print_with_color(:light_magenta, "\n * The whole model is rejected! i.e., Y | (all variables) is not invariant.\n", bold=true)
    elseif isempty(result.S)
        # no causal variable found
        print_with_color(:light_magenta, "\n * Found no causal variable (empty intersection).\n", bold=true)
    else
        print_with_color(:green, "\n * Causal variables include: $(result.S)\n", bold=true)
        if result.selection_only
            print_with_color(:light_blue, " * No confidence intervals produced in selection_only mode\n")
        else
            println(@sprintf("\n%-10s \t %-3s %% \t\t %-3s %%", "variable", result.α * 100, (1 - result.α) * 100))
            for i in result.S
                println(@sprintf("%-10s \t %-4.4f \t %-4.4f", i, result.confint[i, 1], result.confint[i, 2]))
            end
        end
    end
    if length(result.variables_considered) == result.p
        println("\n ⋅ Variables considered include 1:$(result.p)")
    else
        println("\n ⋅ Variables considered include $(result.variables_considered)")
    end
end

"""
    causalSearch(X, y, env, [ S=1:size(X,2) ];
                 α=0.01, method="chow", p_max=8, verbose=true, selection_only=false, n_max_for_exact=5000, max_num_true_causes=length(S))

Searching over subsets in `X[,S]` for direct causes of `y`

# Arguments
* `X`:                 either an n x p matrix or a `DataFrames.DataFrame`
* `y`:                 vector of n
* `env`:               environment indicators (rows of X): 1, 2, ...
* `S`:                 set of variables (col indices of X) to search, can be a Vector or a Set
* `α`:                 significance level (e.g. 0.01)
* `p_max`:             maximum number of variables to consider. 
                       will use lasso (glmnet) to screen out `p_max` number of variables if `p_max < |S|`
* `method`:            
     + `"chow"` for Gaussian linear regression, combined two-sample chow test
     + `"LR-logistic"` for logistic regression (`y` consists of 0 and 1), combined likelihood-ratio test
* `verbose`:           if true, will print each subset tested
* `selection_only`:    if true, will prune supersets of an invariant set;
                       but not able to produce valid confidence intervals
* `n_max_for_exact`:   maximum number of observations of an environment for exact testing; 
                      otherwise a subsample of n_max rows will be used
* `max_num_true_causes`: maximum number of true causal variables; if specified to smaller than `|S|`, 
                        it will skip testing subsets with bigger size than `max_num_true_causes`.
"""
function causalSearch(X::Union{Matrix{Float64}, DataFrame}, y::Vector{Float64}, env::Vector{Int64}, S=1:size(X,2);
                      α=0.01, method="chow", verbose=true, p_max=8,
                      selection_only=false, n_max_for_exact=5000, max_num_true_causes=length(S))
    if method=="chow"
        model = "linear"
        if isa(X, DataFrame)
            X = Matrix{Float64}(X)  # note: current linear fitting has to work with Matrix{Float64}
        end
    elseif method=="LR-logistic"
        model = "logistic"
        # combine into a DataFrame (note: GLM.jl has to work with DataFrame)
        df = DataFrame(hcat(X, y))
        target = names(df)[end]  # target is the last column
    else
        error("method must be one of: `logistic`, `LR-logistic`")
    end
    S = collect(S)
    S = unique(S)
    p = size(X, 2)
    if p_max < length(S)
        @assert model=="linear" "screening unsupported for GLM"
        q = length(S)
        S = S[screen_lasso(X[:, S], y, p_max)]
        print_with_color(:blue, "$p_max variables are screened out from $q variables with lasso: $S\n")
    end
    variables_considered = S[:]
    if max_num_true_causes < length(S)
        print_with_color(:blue, "the size of |S| is restricted to ≦ $max_num_true_causes \n")
    else
        max_num_true_causes = length(S)
    end
    accepted_sets = Dict{Union{Vector{Symbol},Vector{Int64}}, Float64}()
    n_tested_sets = 0
    running_intersection = S
    running_confintervals = zeros(p, 2)
    running_confintervals[:, 1] = Inf
    running_confintervals[:, 2] = -Inf
    n_env = maximum(env)
    if method=="chow"
        for i in 1:n_env
            ni = sum(env.==i)
            if ni > n_max_for_exact
                print_with_color(:blue, @sprintf "environment %d has %d obs, subsample of %d is used\n" i ni n_max_for_exact)
            end
        end
    end
    println(@sprintf "Causal invariance search across %d environments with at α=%s (|S| = %d, method = %s, model = %s)\n" n_env α length(S) method model)    
    max_num_true_causes < length(S) && print_with_color(:blue, "|S| is restricted to subsets with size ≦ $max_num_true_causes.\n")
    # priority queue: S -> -p.value (so sets with higher p-values are tested sooner)
    candidate_sets = PriorityQueue{Union{Set{Int64}, Set{Symbol}}, Float64}()
    enqueue!(candidate_sets, Set{typeof(S[1])}(), 0.)
    size_current_set = 0    
    while size_current_set <= max_num_true_causes
        base_sets = Dict{Union{Set{Int64}, Set{Symbol}}, Float64}()  # set -> (-p_value) 
        while !isempty(candidate_sets)
            # dequeue the set with highest p-value
            _S, neg_p_value = dequeue_pair!(candidate_sets)
            _S_vec = collect(_S)  # convert from Set{Int64} to Vector{Int64}
            # skip supersets under `selection_only`
            selection_only && length(running_intersection)!=length(S) && issubset(running_intersection, _S) && continue
            # test conditional invariance
            n_tested_sets += 1
            if method == "chow"
                rej, p_value, conf_intervals = conditional_inv_test_chow(X[:,_S_vec], y, env, n_env, α=α, n_max_for_exact=n_max_for_exact)
            elseif method == "LR-logistic"
                # target is the last column of df
                rej, p_value, conf_intervals = conditional_inv_test_logistic_LR(df, target, _S_vec, env, n_env, α=α, add_intercept=true)
            end
            if !rej
                # _S is an invariant set
                accepted_sets[_S_vec] = p_value
                # running ∩
                running_intersection = intersect(running_intersection, _S_vec)
                # running ∪
                if isa(_S_vec, Vector{Int64})
                    running_confintervals[_S_vec, 1] = min.(running_confintervals[_S_vec, 1], conf_intervals[:, 1])
                    running_confintervals[_S_vec, 2] = max.(running_confintervals[_S_vec, 2], conf_intervals[:, 2])
                else
                    _idx_vec = [df.colindex[z] for z in _S_vec]
                    running_confintervals[_idx_vec, 1] = min.(running_confintervals[_idx_vec, 1], conf_intervals[:, 1])
                    running_confintervals[_idx_vec, 2] = max.(running_confintervals[_idx_vec, 2], conf_intervals[:, 2])
                end
                if verbose
                    println(@sprintf "S = %-40s: p-value = %-1.4f [%1s] ⋂ = %s" (isempty(_S_vec) ? "[]" : _S_vec) p_value (rej ? " " : "*") running_intersection)
                end
                if isempty(running_intersection)
                    break   # empty, terminate
                end
            else
                if verbose
                    println(@sprintf "S = %-40s: p-value = %-1.4f [%1s] ⋂ = %s" (isempty(_S_vec) ? "[]" : _S_vec) p_value (rej ? " " : "*") running_intersection)
                end
                base_sets[_S] = -p_value  
            end
        end
        isempty(running_intersection) && break  # have to break twice
        # generate sets with size ++     
        for (_base_S, _neg_p_value) in base_sets
            for i in setdiff(S, _base_S)
                S_new = union(_base_S, Set([i]))
                # skipping supersets of running intersection in `selection_only` mode
                selection_only && length(running_intersection)!=length(S) && issubset(running_intersection, S_new) && continue
                if haskey(candidate_sets, S_new)
                    candidate_sets[S_new] = min(candidate_sets[S_new], _neg_p_value)
                else
                    candidate_sets[S_new] = _neg_p_value
                end
            end
        end            
        size_current_set += 1
    end
    println("\nTested $n_tested_sets sets: $(length(accepted_sets)) sets are accepted.")
    if isempty(accepted_sets)
        # model rejected
        return CausalSearchResult(α, p, variables_considered; selection_only = selection_only)
    elseif selection_only
        return CausalSearchResult(running_intersection, α, p, variables_considered; selection_only = true)
    else
        return CausalSearchResult(running_intersection, running_confintervals, α, p, variables_considered; selection_only = false)
    end
end

function causalSearch(df::DataFrame, target::Union{Int64, Symbol}, env::Vector{Int64};
                      α=0.01, method="chow", verbose=true, p_max=8,
                      selection_only=false, n_max_for_exact=5000, max_num_true_causes=Inf)
    if isa(target, Int64)
        target = names(df)[target]
    end
    S = setdiff(names(df), [target])
    X = df[:, S]
    y = df[target]
    causalSearch(X, y, env, S, α=α, method=method, verbose=verbose, p_max=p_max,
                 selection_only=selection_only, n_max_for_exact=n_max_for_exact, max_num_true_causes=max_num_true_causes)
end