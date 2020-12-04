## Causal Inference with Invariant Prediction

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Build Status](https://travis-ci.org/richardkwo/InvariantCausal.jl.svg?branch=master)](https://travis-ci.org/github/richardkwo/InvariantCausal.jl) [![Coverage Status](https://coveralls.io/repos/github/richardkwo/InvariantCausal.jl/badge.svg?branch=master)](https://coveralls.io/github/richardkwo/InvariantCausal.jl?branch=master)

![college](docs/college.png)

This is a **Julia 1.x** implementation for the **Invariant Causal Prediction** algorithm of [Peters, Bühlmann and Meinshausen](https://doi.org/10.1111/rssb.12167). The method uncovers direct causes of a target variable from datasets under different environments (e.g., interventions or experimental settings). 

See also this [R package](https://cran.r-project.org/package=InvariantCausalPrediction) and [this report](docs/InvariantCausal.pdf).

#### Changelog

- 2020/12/03: version 1.0.0 (Julia 1.x)
- 2018/06/20: version 0.1.1 (Julia 0.6)

#### Dependencies

[DataStructures.jl](https://github.com/JuliaCollections/DataStructures.jl), [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl), [GLM.jl](https://github.com/JuliaStats/GLM.jl), [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl), [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) (for lasso screening and requires `gfortran`) and [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

### Installation

Install the package via typing the following in Julia REPL.

```julia
julia> using Pkg
julia> Pkg.add("InvariantCausal")
```

Alternatively, you can install the latest from GitHub.

```Julia
julia> Pkg.add(url="https://github.com/richardkwo/InvariantCausal.git")
```

Use the following to run a full test.

```julia
julia> using InvariantCausal
julia> InvariantCausal._test_full()
```

### Quick Start

Generate a simple [Gaussian structure equation model](https://en.wikipedia.org/wiki/Structural_equation_modeling?oldformat=true) (SEM) with random graph with 21 variables and average degree 3. Note that we assume the SEM is acyclic. The model can be represented as `X = B X + ϵ` with zeros on the diagonals of B (no self-loop). `ϵ` is a vector of independent Gaussian errors. For a variable `i`, variables `j` with coefficients `B[i,j]` non-zero are called the direct causes of `i`. We assume `B` is sparse, and its sparsity pattern is visualized with [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

```julia
julia> using InvariantCausal
julia> using Random
julia> Random.seed!(77)
julia> sem_obs = random_gaussian_SEM(21, 3)

Gaussian SEM with 21 variables:
B =
      Sparsity Pattern
      ┌───────────┐
    1 │⠀⠠⠀⠀⢐⠀⠀⠄⠀⢔⠀│ > 0
      │⠠⠀⠠⠨⠁⠀⠄⠀⠀⠸⠀│ < 0
      │⠠⠈⠈⠀⠌⠠⠀⠅⠀⠩⠉│
      │⠠⣨⠴⠰⠪⠠⠄⠀⠸⠉⣐│
      │⢀⠲⠈⢠⠠⠀⠀⠂⠀⠲⠁│
   21 │⠀⠐⠀⠀⠠⠠⠀⠀⠀⠔⠀│
      └───────────┘
      1          21
        nz = 70σ² = [1.9727697778060356, 1.1224733663047743, 1.1798805640594814, 1.2625825149076064, 0.8503782631176267, 0.5262963446298372, 1.3835334059064883, 1.788996301274282, 1.759286517329432, 0.842571682652995, 1.713382150423666, 1.4524484793202235, 1.9464648511794784, 1.7729995603828317, 0.7110857327642559, 1.6837378902964577, 1.085405687408806, 1.3069888003095986, 1.3933773717634643, 1.0571823834646068, 1.9187793877731028]
```

Suppose we want to infer the direct causes for the last variables, i.e., 9, 11 and 18.

```julia
julia> causes(sem_obs, 21)
3-element Array{Int64,1}:
  9
 11
 18
```

Firstly, let us generate some observational data and call it **environment 1**.

```julia
julia> X1 = simulate(sem_obs, 1000)
```

Then, we simulate from **environment 2** by performing **do-intervention** on variables 3, 4, 5, 6. Here we set them to fixed random values.

```julia
julia> X2 = simulate(sem_obs, [3,4,5,6], randn(4), 1000)
```

We run the algorithm on **environments 1 and 2**.

```julia
julia> causalSearch(vcat(X1, X2)[:,1:20], vcat(X1, X2)[:,21], repeat([1,2], inner=1000))

8 variables are screened out from 20 variables with lasso: [5, 7, 8, 9, 11, 12, 15, 17]
Causal invariance search across 2 environments with at α=0.01 (|S| = 8, method = chow, model = linear)

S = []                                      : p-value = 0.0000 [ ] ⋂ = [5, 7, 8, 9, 11, 12, 15, 17]
S = [5]                                     : p-value = 0.0000 [ ] ⋂ = [5, 7, 8, 9, 11, 12, 15, 17]
S = [17]                                    : p-value = 0.0000 [ ] ⋂ = [5, 7, 8, 9, 11, 12, 15, 17]
S = [15]                                    : p-value = 0.0000 [ ] ⋂ = [5, 7, 8, 9, 11, 12, 15, 17]
S = [12]                                    : p-value = 0.0000 [ ] ⋂ = [5, 7, 8, 9, 11, 12, 15, 17]
S = [11]                                    : p-value = 0.0144 [*] ⋂ = [11]
S = [9]                                     : p-value = 0.0000 [ ] ⋂ = [11]
S = [8]                                     : p-value = 0.0000 [ ] ⋂ = [11]
S = [7]                                     : p-value = 0.0000 [ ] ⋂ = [11]
S = [11, 5]                                 : p-value = 0.0000 [ ] ⋂ = [11]
S = [11, 12]                                : p-value = 0.0000 [ ] ⋂ = [11]
S = [11, 15]                                : p-value = 0.0007 [ ] ⋂ = [11]
S = [7, 11]                                 : p-value = 0.0082 [ ] ⋂ = [11]
S = [11, 8]                                 : p-value = 0.0000 [ ] ⋂ = [11]
S = [9, 11]                                 : p-value = 0.0512 [*] ⋂ = [11]
S = [17, 11]                                : p-value = 0.0000 [ ] ⋂ = [11]
S = [9, 12]                                 : p-value = 0.0000 [ ] ⋂ = [11]
S = [9, 15]                                 : p-value = 0.0064 [ ] ⋂ = [11]
S = [7, 9]                                  : p-value = 0.0000 [ ] ⋂ = [11]
S = [9, 8]                                  : p-value = 0.0000 [ ] ⋂ = [11]
S = [9, 5]                                  : p-value = 0.7475 [*] ⋂ = Int64[]

Tested 21 sets: 3 sets are accepted.

 * Found no causal variable (empty intersection).

 ⋅ Variables considered include [5, 7, 8, 9, 11, 12, 15, 17]
```

The algorithm **cannot find any** direct causal variables (parents) of variable 21 due to **insufficient power** of two environments. The algorithm tends to **discover more** with **more environments**. Let us define a new environment where we perform a **noise (soft) intervention** that changes the equations for 5 variables other than the target. Note it is important that the **target** is left **untouched**.

```Julia
julia> sem_noise, variables_intervened = random_noise_intervened_SEM(sem_obs, p_intervened=5, avoid=[21])

(Gaussian SEM with 21 variables:
B =
      Sparsity Pattern
      ┌───────────┐
    1 │⠀⠠⠀⠀⢐⠀⠀⠄⠀⢔⠀│ > 0
      │⠠⠀⠠⠨⠁⠀⠄⠀⠀⠸⠀│ < 0
      │⠠⠈⠈⠀⠌⠠⠀⠅⠀⠩⠉│
      │⠠⣨⠴⠰⠪⠠⠄⠀⠸⠉⣐│
      │⢀⠲⠈⢠⠠⠀⠀⠂⠀⠲⠁│
   21 │⠀⠐⠀⠀⠠⠠⠀⠀⠀⠔⠀│
      └───────────┘
      1          21
        nz = 70σ² = [1.9727697778060356, 1.1224733663047743, 1.1798805640594814, 1.2625825149076064, 0.8503782631176267, 0.5262963446298372, 1.3835334059064883, 1.788996301274282, 1.759286517329432, 0.5837984015051159, 3.01957479564807, 0.9492838187140921, 1.9398913901673531, 1.7729995603828317, 0.7110857327642559, 1.6837378902964577, 1.2089053651343495, 1.3069888003095986, 1.3933773717634643, 1.0571823834646068, 1.9187793877731028], [17, 13, 10, 11, 12])
```

Here the equations for variables 17, 13, 10, 11, 12 have been changed. Now we simulate from this modified SEM and call it **environment 3**. We run the algorithm on all **3 environments**.

```Julia
julia> X3 = simulate(sem_noise, 1000)
julia> causalSearch(vcat(X1, X2, X3)[:,1:20], vcat(X1, X2, X3)[:,21], repeat([1,2,3], inner=1000))
```

The algorithm searches over subsets for a while and successfully **discovers** variables 11. The other two causes, 9 and 18, can hopefully be discovered given even more environments.

```
causalSearch(vcat(X1, X2, X3)[:,1:20], vcat(X1, X2, X3)[:,21], repeat([1,2,3], inner=1000))
8 variables are screened out from 20 variables with lasso: [4, 5, 7, 8, 9, 11, 12, 16]
Causal invariance search across 3 environments with at α=0.01 (|S| = 8, method = chow, model = linear)

S = []                                      : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [4]                                     : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [16]                                    : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [12]                                    : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [11]                                    : p-value = 0.0084 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [9]                                     : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [8]                                     : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [7]                                     : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [5]                                     : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [4, 11]                                 : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [11, 5]                                 : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [11, 8]                                 : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [7, 11]                                 : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [9, 11]                                 : p-value = 0.0000 [ ] ⋂ = [4, 5, 7, 8, 9, 11, 12, 16]
S = [16, 11]                                : p-value = 0.0709 [*] ⋂ = [11, 16]
S = [11, 12]                                : p-value = 0.0000 [ ] ⋂ = [11, 16]
																			...
S = [7, 9, 4, 16, 11, 5, 12]                : p-value = 0.0000 [ ] ⋂ = [11]
S = [7, 9, 4, 16, 11, 8, 12]                : p-value = 0.0001 [ ] ⋂ = [11]
S = [7, 4, 9, 16, 11, 5, 8, 12]             : p-value = 0.0002 [ ] ⋂ = [11]

Tested 256 sets: 6 sets are accepted.

 * Causal variables include: [11]

   variable   	 1.0 % 		 99.0 %
   11         	 0.1123 	 1.1017

 ⋅ Variables considered include [4, 5, 7, 8, 9, 11, 12, 16]
```

### Functionalities

- The main algorithm `causalSearch(X, y, env, [S]; α=0.01, method="chow", screen="auto", p_max=8, verbose=true, selection_only=false, n_max_for_exact=5000)` 
  - Performs screening if number of covariates exceeds `p_max`
    - `screen="auto"`: `"HOLP"` when p > n, `"lasso"` otherwise
    - `screen="HOLP"`: [High dimensional ordinary least squares projection for screening variables](https://doi.org/10.1111/rssb.12127) when p ≧ n
    - `screen="lasso"`: lasso solution path from `glmnet`
  - Skips supersets of an accepted set under `selection_only = true`, but confidence intervals are not reported
  - When sample size exceeds `n_max_for_exact`, sub-sampling is used for Chow test
- Methods
  - `method="chow"`: Chow test for linear regression
  - `method="logistic-LR"`: likelihood-ratio test for logistic regression
  - `method="logistic-SF"`: [Sukhatme-Fisher test](http://www.jstor.org/stable/2286870) for testing equal mean and variance of logistic prediction residuals
- SEM utilities: `random_gaussian_SEM`, `random_noise_intervened_SEM`, `simulate`, `causes` and `cov` for generating random SEM (Erdos-Renyi), simulation and interventions.
- Variables screening:
  - Lasso (with `glmnet`): `screen_lasso(X, y, pmax)`

###  Features

- High performance implementation in Julia v1.x
- Faster search: 
  - skipping testing supersets of A if A is accepted ( under  `selection_only` mode)
  - Priority queue to prioritize testing sets likely to be invariant

