[![DOI](https://zenodo.org/badge/633839956.svg)](https://zenodo.org/badge/latestdoi/633839956)

# MolStructSampling.jl

This is a package for
- evaluating the joint probability density value of molecular wavefunctions written in a basis of complex explicitly correlated Gaussians,
- drawing random samples using Metropolis&ndash;Hastings Monte Carlo
- and analyzing the resulting sample.

Please see [our publication](https://doi.org/10.26434/chemrxiv-2023-mrxng) describing the research based on this package.

Installation:
- You need [Julia](https://julialang.org/downloads/). On Linux systems, installing it is as easy as downloading the `.tar.gz` file of the desired release, extracting it, and copying its contents to a location that is in your PATH (e.g. `/usr/local/`).
- Start the Julia REPL by typing `julia` in a terminal, then enter Pkg mode by pressing the `]` key. Install MolStructSampling.jl by entering `add https://github.com/LucasLang/MolStructSampling.jl.git`.

