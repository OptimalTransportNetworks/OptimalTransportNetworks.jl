using Documenter
using OptimalTransportNetworks

makedocs(
    sitename = "Optimal Transport Networks Documentation",
    modules = [OptimalTransportNetworks],
    checkdocs = :none,
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)