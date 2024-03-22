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

## Running the analysis

From the root directory of the repository,
run

    make

Plots and tables will be output in
the `figs` and `tabs` directories respectively.

## Reproducibility and reusability

Some aspects of this analysis
make use of `scipy.optimize.curve_fit` and `.minimize`;
as such,
results are not bitwise reproducible
between different CPU architectures.

The code included in this repository
was written with
the analysis of the specific dataset above in mind;
as such,
it may not readily generalise to other data
without additional work.

[dr]: https://doi.org/10.5281/zenodo.10819721
