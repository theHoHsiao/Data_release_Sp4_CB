# Data release for Lattice investigations of the chimera baryon spectrum in the Sp(4) gauge theory

This repository contains the code used to prepare the plots and results
included in
[Lattice investigations of the chimera baryon spectrum in the Sp(4) gauge theory[2311.14663]](https://arxiv.org/abs/2311.14663).
It makes use of data from the corresponding [data release on Zenodo][dr].

## Requirements

- Python 3.11
- GNU Make

## Setup

### Environment

All required Python packages
will be automatically installed into a Python virtual environment by `make`.
This requires an internet connection.
If you plan to run the workflow on a cluster node without internet access,
first prepare the environment by running

    make venv

### Data

To run the full analysis end to end, two sets of data are required.

- Download the file `raw_data.zip` from [the data release][dr],
  and unzip it into the `raw_data` directory.
- Download the file `metadata.zip` from [the data release][dr],
  and unzip it into the `metadata` directory.

For convenience,
we also provide intermediary files to skip
the most computationally parts of the analysis.

- To avoid the step of collating the raw data into HDF5 format,
  download the file `data.h5` from [the data release][dr]
  and place it into the `tmp_data` directory.
  In this case, `raw_data.zip` is not needed.
- To skip the Akaike Information Criterion-based analysis
  (which takes around 600 core hours on an x86-64 cluster),
  download the file `FIT_mass.csv` from [the data release][dr]
  and place it into the `CSVs` directory.
  To ensure that `make` does not try to recompute it,
  use `touch` to set its modification date to be in the future;
  for example,

      touch -A 120000 CSVs/fit_mass.csv

## Running the analysis

From the root directory of the repository,
run

    make

Plots and tables will be output in
the `figs` and `tabs` directories respectively.

### Viewing the 3D plot

As 3D plots can be significantly easier to interpret
if they can be interacted with,
we provide a method to do this.
After running the full analysis described above,
the commands

    source venv/bin/activate
    python generate/plot_figs.py show_3d

will regenerate all plots,
and additionally display the 3D spectrum plot interactively,
so that it may be panned, zoomed, and rotated.

## Reproducibility and reusability

Some aspects of this analysis
make use of `scipy.optimize.curve_fit` and `.minimize`;
as such,
results are not bitwise reproducible
between different CPU architectures.
Additionally,
some aspects
(most notably `analysis/analysis_AIC.py`)
are,
for reasons we do not fully understand,
slightly non-deterministic even on the same hardware,
giving results that differ in the 10th significant figure or beyond.
This is well beyond the precision we consider in our analysis,
and does not affect our conclusions.

The code included in this repository
was written with
the analysis of the specific dataset above in mind;
as such,
it may not readily generalise to other data
without additional work.

[dr]: https://doi.org/10.5281/zenodo.10819721
