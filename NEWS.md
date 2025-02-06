# 0.1.9

* The exact dual solution for the Hessian with `cross_good_congestion = true` does not have good numerical properties in some cases. Therefore, by default now an approximate solution is used which works better for most problems. Users can set `duality = 2` to use the exact solution in the CGC case. 

# 0.1.8

* Fixed dual solution with `cross_good_congestion = true`, and set the default `duality = true` in `init_parameters()`. It is highly recommended to keep `beta <= 1` in fixed labor cases to harness the dual solutions, which yield a tremendous speedup.

# 0.1.7

* If `duality = true`, in `init_parameters()` (and `labor_mobility = false` - no dual solutions available for mobile labor cases), a new direct dual implementation of the model is used (with hard-coded sparse hessians passed directly to Ipopt) which is significantly faster than any other means of solving the model (10x speedup). However, with `cross_good_congestion = true`, the dual solution may be inaccurate. This may be fixed in the future.  

# 0.1.6

## Breaking Changes
* All spatial parameters, including `Lj`, `Lr`, `Hj`, `hj`, `Zjn`, `omegaj`, `omegar`, and `region` are now stored in the `graph` structure created by `create_graph()`. `create_graph()` therefore only returns the `graph` structure, instead of both the (updated) parameters and the graph. Converesely, `init_parameters()` only contains parameters that are independent of the particular geography defined by `create_graph()`.

## Improvements
* Minor improvements to Simulated Annealing.
* Better spline options for plotting frictions surface (geography).
* More faithful translation of `apply_geography()`. 

# 0.1.5

* Removed the MATLAB toolbox and corresponding documentation (PDF files) from the repo to decrease size. A new repo was created for the MATLAB toolbox at [SebKrantz/OptimalTransportNetworkToolbox](https://github.com/SebKrantz/OptimalTransportNetworkToolbox). This repo and especially the `docs` folder continue to be very useful for Julia users, but are no longer part of the Julia library. 