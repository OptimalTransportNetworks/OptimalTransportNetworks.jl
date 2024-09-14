# 0.1.6
## Breaking Changes
* All spatial parameters, including `Lj`, `Lr`, `Hj`, `hj`, `Zjn`, `omegaj`, `omegar`, and `region` are now stored in the `graph` structure created by `create_graph()`. `create_graph()` therefore only returns the `graph` structure, instead of both the (updated) parameters and the graph. Converesely, `init_parameters()` only contains parameters that are independent of the particular geography defined by `create_graph()`.

## Improvements
* Minor improvements to Simulated Annealing.
* Better spline options for plotting frictions surface (geography).
* More faithful translation of `apply_geography()`. 


# 0.1.5

* Removed the MATLAB toolbox and corresponding documentation (PDF files) from the repo to decrease size. A new repo was created for the MATLAB toolbox at [SebKrantz/OptimalTransportNetworkToolbox](https://github.com/SebKrantz/OptimalTransportNetworkToolbox). This repo and especially the `docs` folder continue to be very useful for Julia users, but are no longer part of the Julia library. 