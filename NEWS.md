# 0.3.0

* **Direct-Ipopt primal solvers are now the default for Armington cases (≤1 good per location).** Previously only the dual cases (fixed labor, `beta <= 1`) bypassed JuMP. Now every Armington primal case is solved by a hand-coded `solve_allocation_*` function that calls Ipopt's C interface directly (like the dual solvers), with analytic sparse gradients/Jacobians/Hessians — faster than JuMP. New solvers: `solve_allocation_mobility(_cgc)`, `solve_allocation_partial_mobility(_cgc)`, `solve_allocation_cgc`, `solve_allocation_primal` (fixed labor, `beta > 1`).

* **JuMP is now an opt-in fallback.** Pass `jump = true` to `init_parameters` to force the JuMP path. The general (non-Armington, multi-good-per-location) case still always uses JuMP.

* Fixed a latent bug in `init_parameters`: the utility-function closures (`u`, `uprime`, `usecond`) captured `rho` before it was zeroed for full labor mobility, so they used `rho` instead of `0` in the mobile-labor case. This only affected reported per-location utility `uj` (not welfare).

* **Performance.** The direct solvers reuse their Ipopt problem object across the outer (network) iterations — building the sparsity structure and the problem once, then re-solving warm-started (primal *and* dual) with only the updated `kappa`, exactly as the JuMP path reuses its model. Combined with the analytic derivatives this makes them faster than JuMP across all cases at realistic sizes — e.g. on an 81-node graph (30 fixed iterations): full mobility ~3.8×, fixed-labor `beta>1` ~9×, partial mobility ~10×, fixed cgc ~15×, mobility+cgc ~64×; the gap widens with graph size. The smallest non-cgc problems are at rough parity with JuMP.

* **Manifest regenerated for Julia ≥ 1.12.** The previous `Manifest.toml` was generated under Julia 1.8.5 and pinned old `Ipopt_jll`/`OpenBLAS` binaries (`libopenblas 0.3.17`) that segfault on larger problems under newer Julia — affecting the JuMP path too. The Manifest now resolves Ipopt 1.15 / OpenBLAS 0.3.33, which fixes the crashes. Coin-HSL `ma57`/`ma86` (see README) remain recommended for the best performance: `param[:optimizer_attr] = Dict(:hsllib => HSL_jll.libhsl_path, :linear_solver => "ma57")`.

# 0.2.0

* The reason for the less than ideal numerical properties of the exact dual solution for the Hessian with `cross_good_congestion = true` in v0.1.9 was that the sparse hessian had too few elements. This release fixes the problem by adding some additional off-diagonal elements to the sparse hesssian. The heuristic algorithm is now removed as the exact one always gives better solves (`duality = true` and `duality = 2` both call the exact algorithm now). 

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