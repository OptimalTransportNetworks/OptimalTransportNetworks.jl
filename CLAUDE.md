# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`OptimalTransportNetworks.jl` is a Julia package implementing the quantitative spatial economic model of Fajgelbaum & Schaal (2020, *Econometrica*): it computes the welfare-maximizing transport network on a graph given economic fundamentals (population, productivity per good, housing endowment, optional pre-existing network). It is a JuMP-based translation of the authors' MATLAB `OptimalTransportNetworkToolbox` (v1.0.4b). The reference MATLAB code lives under `misc/matlab/` and is the source of truth when porting or debugging numerical behavior.

## Commands

This is a registered Julia package. From the repo root, start Julia with `julia --project=.` (or `julia --project=examples` to use the examples environment, which adds Revise, BenchmarkTools, GR, Optim, etc.).

```julia
# Develop / load the package
using Pkg; Pkg.instantiate()        # first time, installs deps from Manifest.toml
using OptimalTransportNetworks       # package is the project

# Iterate on source without restarting (examples env has Revise)
using Revise, OptimalTransportNetworks
```

- **Run an example:** `julia --project=examples examples/example01.jl` (examples `01`–`04` are tutorials; `paper_example01`–`04` reproduce paper figures).
- **Build docs:** `julia --project=docs docs/make.jl` (Documenter.jl; output in `docs/build/`).
- **Tests:** there is **no `test/` suite**. The examples serve as the de-facto integration tests — run them to verify changes end-to-end. The package builds figures, so verify by inspecting plots / `results[:welfare]`, not by assertions.

## Core data model

Two user-facing containers, both plain `Dict{Symbol,Any}` while the user edits them, converted internally to `NamedTuple` (`dict_to_namedtuple` in `src/main/helper.jl`):

- **`param`** (from `init_parameters`): non-spatial model parameters only — `alpha, beta, gamma, sigma, rho, a, K`, the `mobility`/`cong` switches, solver options, and closures for utility/production functions.
- **`graph`** (from `create_graph`): both the graph topology **and** all spatial data — `J` (nodes), `nodes`/`adjacency`, `delta_i` (build cost) / `delta_tau` (traversal cost) per edge, and spatial parameters `Zjn` (productivity J×N), `Lj`/`Lr` (labor), `Hj`/`hj` (housing), `omegaj`. Note: as of v0.1.6 all spatial parameters live in `graph`, not `param` (see `NEWS.md`).

`apply_geography` optionally rescales `delta_i`/`delta_tau` for terrain/obstacles.

## Solver pipeline (the big picture)

`optimal_network(param, graph)` in `src/main/optimal_network.jl` is the entry point. It solves the **outer** network-design problem by fixed-point iteration over infrastructure `I`, each iteration solving the **inner** allocation (general-equilibrium trade) problem:

1. Convert dicts → NamedTuples; `represent_edges(graph)` builds the edge incidence representation (`A`, `Apos`, `Aneg`, `edge_start`, `edge_end`).
2. `create_auxdata(param, graph, edges, I)` computes `kappa = I^gamma / delta_tau` (edge capacities) and the reduced `kappa_ex` vector.
3. `get_model(auxdata)` **dispatches** to one of the inner-problem solvers (see below), returning `(model, recover_allocation)`.
4. Loop: solve inner problem → from prices×flows compute the FOC-optimal `I1` → rescale to budget `K` and bounds (`rescale_network!`) → damped update `I0 = 0.5*I0 + 0.5*I1` → push new `kappa` into the model. Repeat until `max|I1-I0|/K_infra < tol`.
5. If `gamma > beta` and `param.annealing`, refine the (non-convex) solution via `annealing` (`src/main/annealing.jl`, simulated annealing over network topology).

Returns a `Dict` keyed by symbols; always includes at least `:welfare`, `:Pjn`, `:PCj`, `:Qjkn`, and `:Ijk` (the optimal network).

### Model dispatch (`get_model` in `src/main/helper.jl`)

This is the most important thing to understand before touching `src/models/`. The inner solver is selected by the cross product of three switches:

- **`param.mobility`** ∈ `{0, 0.5, 1}` (fixed / partial / full labor mobility)
- **`param.cong`** (cross-good congestion on/off)
- **Armington vs. general:** auto-detected — `armington = all(sum(graph.Zjn .> 0, dims=2) .<= 1)`, i.e. every node produces ≤1 good.

**Default path is now direct Ipopt for Armington cases.** As of v0.3.0, every Armington primal case is solved by a hand-coded `solve_allocation_*` function that calls **Ipopt's C interface directly** (like the dual solvers), returning `model = nothing` so `optimal_network` takes the `model === nothing` branch. These are faster than JuMP. The files: `solve_allocation_mobility(_cgc)`, `solve_allocation_partial_mobility(_cgc)`, `solve_allocation_cgc`, `solve_allocation_primal` — one per `(mobility × cong)` (the fixed-labor, no-cong, `beta≤1` case stays on the even-faster dual). They share `src/models/solve_allocation_primal_helpers.jl` (`run_ipopt_primal`, sparse-structure + value scatter via `struct_from_triplets`/`fill_values_from_triplets!`, and the `felicity*` Cobb-Douglas helpers).

**JuMP is the fallback**, taken when `param.jump == true` **or** the input is **non-Armington** (a node produces ≥2 goods, which the hand-coded solvers don't support). Then `get_model` builds the matching `model_*` JuMP model (general `model_*` for multi-good-per-node, `*_armington` when `param.jump` forces JuMP on an Armington input). Naming encodes the switches, e.g. `model_partial_mobility_cgc_armington.jl`.

**Special case — duality:** when `mobility == 0` and `beta <= 1` and `param.duality > 0`, `get_model` dispatches to `solve_allocation_by_duality` (or `..._cgc`) — hard-coded sparse gradient/Hessian, the fastest path, handling both Armington and general. This takes precedence over the direct-primal and JuMP branches.

**Direct-solver conventions** (see `git log` / `misc/matlab/solve_allocation_*.m`): full mobility & all cgc cases use **split** directional flows `Qin_direct`/`Qin_indirect`; partial mobility & the fixed primal use a **single signed** `Qin` (split flows find worse local optima in non-convex partial mobility). Each `solve_allocation_*(x0, auxdata, verbose) -> (Dict, status, x)`; prices come from the balanced-flow constraint multipliers (`prob.mult_g`). Validate any change with a finite-difference check of the hand-coded Jacobian/Hessian at a random interior point (Ipopt's `derivative_test` gives false positives at the degenerate `x0=1e-6`).

**Problem reuse (perf).** `optimal_network` threads a per-run `struct_cache::Dict` through `auxdata`. The direct solvers compute their sparsity structure once (`get_structs`/`struct_from_triplets` build a column-major pattern + a linear-index→position map; values are *scattered* into the Ipopt buffers via `fill_values_from_triplets!` — never build a `sparse()` inside a callback, it corrupts the heap), then `run_ipopt_primal` caches the Ipopt problem object keyed by solver and **reuses it across outer iterations**, re-solving warm-started after updating `kappa_ex` in place in the callbacks' captured `saux`. This (plus analytic derivatives) makes the direct path several× to ~60× faster than JuMP at realistic sizes. Annealing builds its own `auxdata` without the cache, so it falls back to per-call construction (correct, just not reused).

**Environment:** the bundled MUMPS/OpenBLAS in old `Ipopt_jll` (pre-Julia-1.12 Manifests) segfaults on larger problems — keep the Manifest resolved for the running Julia (Ipopt ≥ 1.15 / OpenBLAS ≥ 0.3.33).

**Custom models:** users may pass `param[:model]` and `param[:recover_allocation]` to override dispatch entirely.

## Directory map

- `src/main/` — pipeline: `optimal_network.jl`, `annealing.jl`, `create_graph.jl`, `init_parameters.jl`, `apply_geography.jl`, `nodes.jl` (`find_node`/`add_node`/`remove_node`), `plot_graph.jl`, `helper.jl` (dispatch + edge/auxdata machinery).
- `src/models/` — the ~16 `model_*` builders + their `recover_allocation_*` functions, plus the two `solve_allocation_by_duality*` direct-Ipopt solvers.
- `src/OptimalTransportNetworks.jl` — module; auto-includes every `.jl` under `main/` then `models/` via `include_directory`, and lists the exports.
- `examples/` — runnable tutorials and paper reproductions.
- `misc/` — MATLAB reference (`misc/matlab/`), experiments, LaTeX derivations of the duality solution (`misc/duality_*`), figures. Untracked scratch; not part of the package.

## Conventions & gotchas

- Adding a source file: just drop it in `src/main/` or `src/models/` — `include_directory` picks up all `.jl` automatically; no manual `include` needed. Add new public functions to the `export` list in `src/OptimalTransportNetworks.jl`.
- The graph is undirected; edges are stored once with `nodes[j][k] > j` guarding the canonical direction (see `represent_edges`, `kappa_extract`). Respect this when iterating edges.
- `recover_allocation` for any new/custom model **must** return a Dict including `:welfare`, `:Pjn`, `:PCj`, `:Qjkn` — `optimal_network` reads these to compute the next `I`.
- Convexity guards live at the top of `optimal_network`: `a > 1` and (`nu < 1` with congestion) error out; non-convex cases (`gamma > beta`) are where annealing matters.
- For performance, the README documents enabling Coin-HSL linear solvers via `param[:optimizer_attr]` and MathOptSymbolicAD via `param[:model_attr]`.
