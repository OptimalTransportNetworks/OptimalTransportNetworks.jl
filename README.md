# OptimalTransportNetworks.jl
**Optimal Transport Networks in Spatial Equilibrium - in Julia and MATLAB**

Modern Julia ([JuMP](https://github.com/jump-dev/JuMP.jl)) translation of the MATLAB OptimalTransportNetworkToolbox (v1.0.4b) implementing the quantitative spatial economic model of:

Fajgelbaum, P. D., & Schaal, E. (2020). Optimal transport networks in spatial equilibrium. *Econometrica, 88*(4), 1411-1452.

The model/software uses duality principles to optimize over the space of networks, nesting an optimal flows problem and a neoclasical general-equilibrium trade model into a global network design problem to derive the optimal (welfare maximizing) transport network (extension) from any primitive set of economic fundamantals [population per location, productivity per location for each of *N* traded goods, endowment of a non-traded good, and (optionally) a pre-existing transport network]. For more information about the model see [this folder](https://github.com/SebKrantz/OptimalTransportNetworks.jl/tree/main/misc/paper_materials).

The model is the first of its kind and a pathbreaking contribution towards the welfare maximizing planning of transport infrastructure. Its creation has been funded by the European Union through an [ERC Research Grant](https://cordis.europa.eu/project/id/804095). The author of this Julia library has no personal connections to the authors, but has used their Matlab library for research purposes and belives that it deserves an accessible open-source implementation. Community efforts to further improve the code are welcome. In particular, there is a probabilistic extenstion to solving the model using MCMC methods which may be more suitable for large networks, implemented in:

Kreindler, G., Gaduh, A., Graff, T., Hanna, R., & Olken, B. A. (2023). Optimal Public Transportation Networks: Evidence from the World's Largest Bus Rapid Transit System in Jakarta (No. w31369). National Bureau of Economic Research.

## Example

The code for this example is in [example04.jl](https://github.com/SebKrantz/OptimalTransportNetworks.jl/blob/main/examples/example04.jl). See the [examples folder](https://github.com/SebKrantz/OptimalTransportNetworks.jl/blob/main/examples) for more examples.

This plot shows the endowments on a map-graph: circle size is population, circle colour is productivity (the central node is more productive), the black lines indicate geographic barriers, and the background is shaded according to the cost of network building (elevation), indicating a mountain in the upper right corner. 

![](misc/figures/example04_setup.png)

This plot shows the optimal network after 200 iterations, keeping population fixed and not allowing for cross-good congestion. The size of nodes indicates consumption in each node. 

![](misc/figures/example04_solution.png)

## Notes

* The Julia implementation does not provide hard-coded Gradients, Jacobians, and Hessians as the MATLAB implementation does for some model cases, but relies solely on JuMP's automatic differentiation. This has proven ineffective for dual solutions to the model where the objective is quite complex. Thus, at present, duality does not help to speed up computations in Julia, and accordingly the default is `duality = false`. I expect this to change in when [support for detecting nonlinear subexpressions](https://github.com/jump-dev/JuMP.jl/issues/3738) will be added to JuMP.  

* Related, I expect symbolic autodifferentiation via [MathOptSymbolicAD.jl](https://github.com/lanl-ansi/MathOptSymbolicAD.jl) to provide significant performance improvements. Presently this cannot be used on these models because logical or comparison operators are not supported. See [this issue](https://github.com/lanl-ansi/MathOptSymbolicAD.jl/issues/31). Once support is added, the symbolic backend can be activated using:

    ```julia
    import MathOptInterface as MOI
    import MathOptSymbolicAD

    param[:model_attr] = Dict(:backend => (MOI.AutomaticDifferentiationBackend(), 
                                           MathOptSymbolicAD.DefaultBackend())) 
                                    # Or:  MathOptSymbolicAD.ThreadedBackend()
    ```

* It is recommended to use Coin-HSL linear solvers for [Ipopt](https://github.com/jump-dev/Ipopt.jl) to speed up computations. In my opinion the simplest way to use them is do get a (free for academics) license and download the binaries [here](https://licences.stfc.ac.uk/product/coin-hsl), extract them somewhere, and then set the `hsllib` and `linear_solver` options in the `optimizer_attr` dictionary in `param` as follows:

    ```julia
    param[:optimizer_attr] = Dict(:hsllib => "/usr/local/lib/libhsl.dylib", # Adjust path
                                  :linear_solver => "ma57") # Use ma57, ma86 or ma97
    ```
    The [Ipopt.jl README](https://github.com/jump-dev/Ipopt.jl?tab=readme-ov-file#linear-solvers) suggests to use the larger LibHSL package for which there exists a Julia module and proceed similarly. In addition, users may try an [optimized BLAS](https://github.com/jump-dev/Ipopt.jl?tab=readme-ov-file#blas-and-lapack) and see if it yields significant performance gains (and let me know if it does). 