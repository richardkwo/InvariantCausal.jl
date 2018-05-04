## Causal Inference with Invariant Prediction

This is a Julia v.0.6 implementation for the Invariant Causal Prediction algorithm of [Peters, Bühlmann and Meinshausen](https://doi.org/10.1111/rssb.12167). The method uncovers direct causes of a target variable from datasets under different environments (e.g., interventions or experimental settings). 

See also this [R package](https://cran.r-project.org/package=InvariantCausalPrediction).

#### Dependencies

[DataStructures.jl](https://github.com/JuliaCollections/DataStructures.jl), [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl), [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) (for lasso screening) and [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

### Installation

Install the package via

```Julia
Pkg.clone("https://github.com/richardkwo/InvariantCausal.git")
```

#### Test

Run all tests with 

```julia
Pkg.test("InvariantCausal)
```

### Quick Start

Generate a simple Gaussian structure equation model with random graph with 21 variables and average degree 3. 

```julia
julia> using InvariantCausal

julia> sem_obs = random_gaussian_SEM(21, 3)
Gaussian SEM with 21 variables:
B =
      Sparsity Pattern
      ┌───────────┐
    1 │⢀⠀⠀⠀⠀⡢⠀⠄⢤⢀⠄│ > 0
      │⠰⠔⡂⠀⠐⡅⠄⠀⠮⠐⠀│ < 0
      │⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
      │⠐⠀⠑⢠⠌⠄⠀⠀⢀⢈⠐│
      │⠸⠈⠃⠀⠀⠃⣈⡀⡙⠀⠉│
   21 │⠀⠠⠀⠀⠆⠀⠀⠄⠑⠐⠀│
      └───────────┘
      1          21
        nz = 64
σ² = [0.558131, 0.780264, 0.743692, 1.70662, 1.54865, 1.26696, 1.90837, 0.958861, 1.0034, 1.77877, 0.67226, 1.77009, 0.845154, 0.728498, 0.851818, 1.62949, 1.41946, 1.58322, 1.89314, 0.742785, 0.617711]
```

Suppose we want to infer the direct causes for the last variables, which are

```
julia> causes(sem_obs, 21)
3-element Array{Int64,1}:
  3
  8
 14
```

Firstly, let us generate some observational data and call it environment 1.

```julia
julia> X1 = simulate(sem_obs, 500)
```

Then, we simulate from environment 2 by performing do-intervention on variables 3, 4, 5. Here we set them to fixed random values.

```julia
X2 = simulate(sem_obs, [3,4,5], randn(3), 500)
```

We run the algorithm on environments 1 and 2.

```julia
julia> causalSearch(vcat(X1, X2)[:,1:20], vcat(X1, X2)[:,21], repeat([1,2], inner=500))
8 variables are screened out from 20 variables with lasso: [1, 3, 5, 8, 9, 10, 12, 14]
Causal invariance search across 2 environments with at α=0.01 (|S| = 8, method = chow)

S = []                                      : p-value = 0.0006 [ ] ⋂ = [1, 3, 5, 8, 9, 10, 12, 14]
S = [1]                                     : p-value = 0.0010 [ ] ⋂ = [1, 3, 5, 8, 9, 10, 12, 14]
S = [14]                                    : p-value = 0.0169 [*] ⋂ = [14]
S = [12]                                    : p-value = 0.0000 [ ] ⋂ = [14]
S = [10]                                    : p-value = 0.0028 [ ] ⋂ = [14]
S = [9]                                     : p-value = 0.0004 [ ] ⋂ = [14]
S = [8]                                     : p-value = 0.0004 [ ] ⋂ = [14]
S = [5]                                     : p-value = 0.0031 [ ] ⋂ = [14]
S = [3]                                     : p-value = 0.0577 [*] ⋂ = Int64[]

 * Found no causal variable (empty intersection).

 ⋅ Variables considered include [1, 3, 5, 8, 9, 10, 12, 14]
```

The algorithm cannot find a direct cause due to insufficient power. To increase power, we can include data from more environments.



### Features

- High performance implementation in Julia v.0.6
- Faster search: w
  - skipping testing supersets of A if A is accepted ( under  `selection_only` mode)
  - Priority queue to prioritize testing sets likely to be invariant

### Todo

- ~~Confidence intervals~~
- ~~Variable screening~~
  - ~~glmnet~~
- ~~Subsampling for large n in Chow's test~~
- Nonparametric two-sample tests
- Hidden variable case
- ~~Inference of graph and plotting~~

### Issues

- ~~Better reporting~~

