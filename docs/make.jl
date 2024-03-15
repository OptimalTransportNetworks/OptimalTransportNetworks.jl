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

deploydocs(
    repo = "github.com/SebKantz/OptimalTransportNetworks.jl.git",
    # deploy_config = Dict("DOCUMENTER_KEY" => ENV["DOCUMENTER_KEY"]), 
    # push_preview = true, ..
)