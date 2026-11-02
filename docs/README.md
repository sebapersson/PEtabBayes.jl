# Building the documentation

From this directory, set up the Julia environment once:

```sh
julia setup.jl
```

This installs the documentation dependencies and uses the PEtabBayes checkout in
the parent directory. The first run may take several minutes to precompile.

Build the documentation with:

```sh
julia make.jl
```

Both scripts activate this directory's Julia environment automatically.
DocumenterVitepress installs the JavaScript dependencies using its bundled
Node.js, so no separate Node.js installation is needed on Linux. The first build
requires internet access to download dependencies.

The generated site is in `build/1/`. To preview it locally, run:

```sh
julia --project=. -e 'using DocumenterVitepress; DocumenterVitepress.dev_docs("build")'
```

Open the localhost URL printed by VitePress. Re-run `julia make.jl` after changing
the documentation sources or package docstrings. Deployment is skipped outside CI.
