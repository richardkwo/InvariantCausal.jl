## Causal Inference with Invariant Prediction

This is a Julia v.0.6 implementation for the Invariant Causal Prediction algorithm of [Peters, Bühlmann and Meinshausen](https://doi.org/10.1111/rssb.12167). The method uncovers direct causes of a target variable from datasets under different environments (e.g., interventions or experimental settings). 

See also this [R package](https://cran.r-project.org/package=InvariantCausalPrediction).

#### Dependencies

[DataStructures.jl](https://github.com/JuliaCollections/DataStructures.jl), [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl), [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) (for lasso screening).

### Quick Start

Generate a simple Gaussian structure equation model with random graph. 



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

