using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
Pkg.instantiate()
