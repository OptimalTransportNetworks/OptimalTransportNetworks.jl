using Documenter
using OptimalTransportNetworks

DocMeta.setdocmeta!(OptimalTransportNetworks, :DocTestSetup, :(using OptimalTransportNetworks); recursive=true)

makedocs(
    modules = [OptimalTransportNetworks],
    authors = "Sebastian Krantz",
    repo = "https://github.com/SebKrantz/OptimalTransportNetworks.jl/blob/{commit}{path}#{line}",
    sitename = "OptimalTransportNetworks.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://SebKrantz.github.io/OptimalTransportNetworks.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/SebKrantz/OptimalTransportNetworks.jl.git",
    devbranch = "main",
    tag = "stable",
    push_preview = true,
)