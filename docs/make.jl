using Documenter
using OptimalTransportNetworks

makedocs(
    sitename = "OptimalTransportNetworks.jl",
    modules = [OptimalTransportNetworks],
    checkdocs = :none,
    format = Documenter.HTML(
        # Add these options for version switching
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://SebKrantz.github.io/OptimalTransportNetworks.jl/stable"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/SebKrantz/OptimalTransportNetworks.jl.git",
    devbranch = "development",
    push_preview = true,
    versions = ["stable" => "v^", "dev" => "development"]
)