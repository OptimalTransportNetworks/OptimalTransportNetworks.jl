using Documenter
using OptimalTransportNetworks

makedocs(;
    sitename = "OptimalTransportNetworks.jl",
    modules = [OptimalTransportNetworks],
    format = Documenter.HTML(
        # Add these options for version switching
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://OptimalTransportNetworks.github.io/OptimalTransportNetworks.jl"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ],
    checkdocs = :none,
)

deploydocs(; repo="github.com/OptimalTransportNetworks/OptimalTransportNetworks.jl", push_preview=true)
