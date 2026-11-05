using Pkg
Pkg.activate(@__DIR__)

using Documenter
using DocumenterCitations
using DocumenterVitepress
using PEtabBayes

DocMeta.setdocmeta!(PEtabBayes, :DocTestSetup, :(using PEtabBayes); recursive = false)

bib = CitationBibliography(
    joinpath(@__DIR__, "refs.bib"), style = :numeric, show_backlinks = false
)

makedocs(
    modules = [PEtabBayes],
    sitename = "PEtabBayes.jl",
    repo = Remotes.GitHub("sebapersson", "PEtabBayes.jl"),
    authors = "Henrik Häggström and Sebastian Persson",
    checkdocs = :exports,
    warnonly = false,
    plugins = [bib],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/sebapersson/PEtabBayes.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "tutorial.md",
        "API" => "API.md",
        "Contributing" => "contributing.md",
    ],
)

DocumenterVitepress.deploydocs(
    repo = "github.com/sebapersson/PEtabBayes.jl.git",
    target = "build", # this is where Vitepress stores its output
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true,
)
